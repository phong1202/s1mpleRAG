"""Who is asking. Until login exists, always the seeded admin.

Every endpoint that needs a user takes `Depends(get_current_user)`. When login
arrives, only the body of this function changes -- it will read a token or a
cookie -- and the endpoints stay as they are.
"""

import uuid

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.exceptions.base import AppException
from app.exceptions.error_codes import ErrorCode
from app.models import User
from app.utils.database import get_session

# Seeded by the users migration. Fixed, so tests, backfilled rows and logs can
# name it, and it does not depend on the order migrations ran in.
ADMIN_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")


async def get_current_user(session: AsyncSession = Depends(get_session)) -> User:
    # AUTH_MODE=auto_admin, the only mode for now.
    user = await session.get(User, ADMIN_ID)
    if user is None:
        # The migrations have not run. Creating the admin here would hide that.
        raise AppException(ErrorCode.ADMIN_USER_MISSING)
    return user
