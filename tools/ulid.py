import uuid
import time
import os


class GroupId:
    '''Generate a group id with ULID format.
    
    Example:
    >>> print(GroupId().ulid)
    '''
    @property
    def ulid(self):
        randomness = os.urandom(10)
        uuid_ = uuid.UUID(
            bytes=self.time_bytes+randomness,
            version=1
        )

        return str(uuid_)

    @property
    def time_bytes(self):
        time_ = int(time.time()*1000)
        return time_.to_bytes(6, 'big')
