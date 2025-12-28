"""Training scheduler with cron-based scheduling.

This module provides scheduled execution of training jobs
with configurable triggers and monitoring.
"""

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:
    from croniter import croniter
except ImportError:
    croniter = None

logger = logging.getLogger(__name__)


class SchedulerState(Enum):
    """Scheduler state."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


class JobState(Enum):
    """Training job state."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SchedulerConfig:
    """Configuration for the training scheduler."""
    # Cron expression for scheduling (e.g., "0 2 * * *" for 2 AM daily)
    cron_expression: str = "0 2 * * *"
    
    # Minimum samples before triggering training
    min_feedback_count: int = 1000
    
    # Maximum time between training runs (hours)
    max_training_interval_hours: int = 168  # 1 week
    
    # Timezone for cron
    timezone: str = "UTC"
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: int = 300
    
    # Concurrent job limit
    max_concurrent_jobs: int = 1
    
    # Enable/disable
    enabled: bool = True
    
    # Auto-deploy if training succeeds
    auto_deploy: bool = True
    deploy_threshold: float = 0.95  # Must beat baseline by this factor


@dataclass
class ScheduledJob:
    """A scheduled training job."""
    job_id: str
    scheduled_time: datetime
    state: JobState = JobState.PENDING
    
    # Execution details
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Results
    result: Optional[Any] = None
    error: Optional[str] = None
    retries: int = 0
    
    # Trigger information
    trigger_type: str = "scheduled"  # scheduled, manual, threshold
    trigger_details: Dict[str, Any] = field(default_factory=dict)


class TrainingScheduler:
    """Schedules and manages training job execution.
    
    Supports:
    - Cron-based scheduling
    - Feedback count thresholds
    - Manual triggering
    - Job retry with backoff
    - Concurrent job limiting
    """
    
    def __init__(
        self,
        config: SchedulerConfig,
        training_callback: Optional[Callable] = None,
        feedback_counter: Optional[Callable[[], int]] = None,
    ):
        """Initialize scheduler.
        
        Args:
            config: Scheduler configuration
            training_callback: Async function to call for training
            feedback_counter: Function returning current feedback count
        """
        self.config = config
        self.training_callback = training_callback
        self.feedback_counter = feedback_counter
        
        self.state = SchedulerState.STOPPED
        self.jobs: Dict[str, ScheduledJob] = {}
        self.job_history: List[ScheduledJob] = []
        
        self._lock = threading.Lock()
        self._scheduler_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        self.last_training_time: Optional[datetime] = None
        self.last_feedback_count: int = 0
        
        logger.info(
            f"TrainingScheduler initialized with cron={config.cron_expression}, "
            f"min_feedback={config.min_feedback_count}"
        )
    
    def start(self):
        """Start the scheduler."""
        if self.state == SchedulerState.RUNNING:
            logger.warning("Scheduler already running")
            return
        
        if not self.config.enabled:
            logger.info("Scheduler disabled in config")
            return
        
        self._stop_event.clear()
        self.state = SchedulerState.RUNNING
        
        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler_loop,
            daemon=True,
        )
        self._scheduler_thread.start()
        
        logger.info("Scheduler started")
    
    def stop(self):
        """Stop the scheduler."""
        if self.state == SchedulerState.STOPPED:
            return
        
        self._stop_event.set()
        self.state = SchedulerState.STOPPED
        
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)
            self._scheduler_thread = None
        
        logger.info("Scheduler stopped")
    
    def pause(self):
        """Pause the scheduler."""
        if self.state == SchedulerState.RUNNING:
            self.state = SchedulerState.PAUSED
            logger.info("Scheduler paused")
    
    def resume(self):
        """Resume the scheduler."""
        if self.state == SchedulerState.PAUSED:
            self.state = SchedulerState.RUNNING
            logger.info("Scheduler resumed")
    
    def trigger_now(self, reason: str = "manual") -> str:
        """Manually trigger a training job.
        
        Args:
            reason: Reason for the manual trigger
            
        Returns:
            Job ID
        """
        job = self._create_job(
            trigger_type="manual",
            trigger_details={"reason": reason},
        )
        
        # Run in background thread
        thread = threading.Thread(
            target=self._execute_job_sync,
            args=(job,),
        )
        thread.start()
        
        logger.info(f"Manually triggered job {job.job_id}")
        return job.job_id
    
    def get_next_run_time(self) -> Optional[datetime]:
        """Get the next scheduled run time."""
        if croniter is None:
            logger.warning("croniter not installed")
            return None
        
        try:
            cron = croniter(self.config.cron_expression, datetime.utcnow())
            return cron.get_next(datetime)
        except Exception as e:
            logger.error(f"Failed to parse cron expression: {e}")
            return None
    
    def get_job_status(self, job_id: str) -> Optional[ScheduledJob]:
        """Get status of a specific job."""
        with self._lock:
            return self.jobs.get(job_id)
    
    def get_active_jobs(self) -> List[ScheduledJob]:
        """Get all active (pending or running) jobs."""
        with self._lock:
            return [
                job for job in self.jobs.values()
                if job.state in (JobState.PENDING, JobState.RUNNING)
            ]
    
    def get_job_history(self, limit: int = 100) -> List[ScheduledJob]:
        """Get recent job history."""
        return self.job_history[-limit:]
    
    def should_train(self) -> Tuple[bool, str]:
        """Check if training should be triggered.
        
        Returns:
            Tuple of (should_train, reason)
        """
        # Check if already running max jobs
        active_jobs = self.get_active_jobs()
        if len(active_jobs) >= self.config.max_concurrent_jobs:
            return False, "max_concurrent_jobs_reached"
        
        # Check feedback count threshold
        if self.feedback_counter:
            current_count = self.feedback_counter()
            new_feedback = current_count - self.last_feedback_count
            
            if new_feedback >= self.config.min_feedback_count:
                return True, f"feedback_threshold_reached ({new_feedback} new)"
        
        # Check max interval
        if self.last_training_time:
            hours_since = (datetime.utcnow() - self.last_training_time).total_seconds() / 3600
            if hours_since >= self.config.max_training_interval_hours:
                return True, f"max_interval_exceeded ({hours_since:.1f}h)"
        
        return False, "no_trigger"
    
    def _run_scheduler_loop(self):
        """Main scheduler loop."""
        logger.info("Scheduler loop started")
        
        while not self._stop_event.is_set():
            try:
                if self.state == SchedulerState.RUNNING:
                    self._check_and_schedule()
                
                # Sleep for a short interval
                self._stop_event.wait(timeout=60)
                
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                time.sleep(5)
        
        logger.info("Scheduler loop stopped")
    
    def _check_and_schedule(self):
        """Check conditions and schedule jobs."""
        # Check cron schedule
        next_run = self.get_next_run_time()
        
        if next_run and next_run <= datetime.utcnow():
            should_train, reason = self.should_train()
            
            if should_train:
                job = self._create_job(
                    trigger_type="scheduled",
                    trigger_details={"reason": reason},
                )
                self._execute_job_sync(job)
            else:
                logger.debug(f"Skipping scheduled training: {reason}")
    
    def _create_job(
        self,
        trigger_type: str,
        trigger_details: Dict[str, Any],
    ) -> ScheduledJob:
        """Create a new scheduled job."""
        import uuid
        
        job = ScheduledJob(
            job_id=str(uuid.uuid4())[:8],
            scheduled_time=datetime.utcnow(),
            trigger_type=trigger_type,
            trigger_details=trigger_details,
        )
        
        with self._lock:
            self.jobs[job.job_id] = job
        
        return job
    
    def _execute_job_sync(self, job: ScheduledJob):
        """Execute a job synchronously."""
        job.state = JobState.RUNNING
        job.started_at = datetime.utcnow()
        
        logger.info(f"Starting job {job.job_id}")
        
        try:
            if self.training_callback:
                # Run async callback in event loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(self.training_callback())
                finally:
                    loop.close()
                
                job.result = result
                job.state = JobState.COMPLETED
                
                # Update tracking
                self.last_training_time = datetime.utcnow()
                if self.feedback_counter:
                    self.last_feedback_count = self.feedback_counter()
                
                logger.info(f"Job {job.job_id} completed successfully")
            else:
                logger.warning("No training callback configured")
                job.state = JobState.COMPLETED
            
        except Exception as e:
            job.error = str(e)
            job.retries += 1
            
            if job.retries < self.config.max_retries:
                job.state = JobState.PENDING
                logger.warning(
                    f"Job {job.job_id} failed, will retry "
                    f"({job.retries}/{self.config.max_retries}): {e}"
                )
                time.sleep(self.config.retry_delay_seconds)
                self._execute_job_sync(job)
            else:
                job.state = JobState.FAILED
                logger.error(f"Job {job.job_id} failed after max retries: {e}")
        
        finally:
            job.completed_at = datetime.utcnow()
            self.job_history.append(job)
            
            # Limit history size
            if len(self.job_history) > 1000:
                self.job_history = self.job_history[-500:]


from typing import Tuple

