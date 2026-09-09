from arq.connections import RedisSettings

from backend.config import settings
from backend.worker.tasks import process_images, startup


class WorkerSettings:
    functions = [process_images]
    on_startup = startup
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_jobs = 4
    job_timeout = 600  # 10 min
