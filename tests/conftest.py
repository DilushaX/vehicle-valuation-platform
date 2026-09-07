import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from database.connection import Base
from database.repository import VehicleRepository


@pytest.fixture
def db_session():
    """
    Provides an isolated in-memory SQLite database session for testing.
    """
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = Session()

    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def vehicle_repo(db_session):
    """
    Provides a VehicleRepository connected to the isolated in-memory database session.
    """
    return VehicleRepository(db_session)
