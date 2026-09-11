from arq.connections import RedisSettings

from backend.config import settings
from backend.worker.tasks import process_images, shutdown, startup


class WorkerSettings:
    functions = [process_images]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 1  # jobs run in threads on shared models; more would multiply memory
    job_timeout = 1800  # 30 min: multi-view on CPU takes minutes for a handful of photos
