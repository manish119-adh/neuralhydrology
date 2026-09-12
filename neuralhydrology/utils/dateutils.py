
from datetime import datetime
import numpy as np

def get_month_year(val):
    # convert to MonthYear
    if isinstance(val, str):
        return MonthYear.parse(val)
    elif isinstance(val, datetime):
        return MonthYear.from_datetime(val)
    elif isinstance(val, np.datetime64):
        return MonthYear.from_numpy_datetime(val)
    elif isinstance(val, np.ndarray):
        # Allow only if it is single value of type datetime or integer
        
        if not (np.issubdtype(val.dtype, np.integer) or np.issubdtype(val.dtype, np.datetime64)):
            raise TypeError("type must be either an integral type or date time type")
        if val.size > 1:
            raise ValueError("Multiple value array cannot be converted to MonthYear")
        val = val.squeeze()
        return get_month_year(val.item())
    elif isinstance(val, MonthYear):
        return val
    elif isinstance(val, np.integer):
        return MonthYear(int(val))
    elif isinstance(val, int):
        return MonthYear(val)
    else:
        raise TypeError(f"Cannot convert from {type(val)} to MonthYear type")

import numpy as np
class MonthYear(int):
    """
    Define a datatype called monthyear
    represented by a single int year *12 + month-1
    """

    def __new__(cls, val):
        instance = super().__new__(cls, val)
        return instance

    @staticmethod
    def new(year, month):
        return MonthYear(year * 12 + month - 1)

    @staticmethod
    def from_datetime(val):
        return MonthYear.new(val.year, val.month)

    @staticmethod
    def from_numpy_datetime(datetime):
        month = str(datetime.astype('datetime64[M]'))
        return MonthYear.parse(month)

    
        

    @staticmethod
    def parse(strval):
        [year, month] = strval.split("-", 1)
        return MonthYear.new(int(year.strip()), int(month.strip()))

    month = property(lambda self: int(self)%12 + 1)
    year = property(lambda self: int(self) // 12)

    val = property(lambda self: int(self))

    def __str__(self):
        return f"{self.year}-{self.month:02}"


    def __repr__(self):
        return f"dateutils.MonthYear({self}):val {self.val}"
        

    def __add__(self, next):
        if isinstance(next, MonthYear):
            # 1 month is represented by value 0
            # so if it is another monthyear we should add by one
            raise TypeError("Two dates cannot be added")
        if not isinstance(next, int):
            raise TypeError("Cannot add non integer")
        return MonthYear(int(self) + next)  

    def __radd__(self, next):
        """
        If next is MonthYear this will not be executed because __add__ 
        is called
        """
        return self.__add__(next)

    def __sub__(self, next):
        """
        Subtracting another MonthYear returns number
        of months differene between them.
        Subtracting an int gives the time that many months in
        the past
        """
        if isinstance(next, MonthYear):
            # If it is another monthyear return the 
            # number of months difference
            return int(self) - int(next)
        if not isinstance(next, int):
            raise TypeError("Cannot subtract non integer")
        return MonthYear(int(self) - next)

    def __mul__(self, next):raise TypeError("Date multiplication undefined")
    def __rmul__(self, next):raise TypeError("Date multiplication undefined")
    def __truediv__(self, next):raise TypeError("Date division undefined")
    def __floordiv__(self, next):raise TypeError("Date division undefined")
    def __mod__(self, next):raise TypeError("Date division undefined")
    def __pow__(self, next):raise TypeError("Date power undefined")
    def __rtruediv__(self, next): raise TypeError("Date division undefined")
    def __rmul__(self, next): raise TypeError("Date division undefined")
    def __rpow__(self, next): raise TypeError("Date power undefined")
    

    def __rsub__(self, next):
        """
        If next is MonthYear __add__ will be given priority
        """
        raise TypeError("MonthYear cannot be subtracted except from another MonthYear")

    
if __name__ == "__main__":
    dt_array = np.array(['2026-09-11', '2024-05-23'], dtype='datetime64[D]')
    months_dt64 = dt_array.astype('datetime64[M]')
    print(str(months_dt64[0]))




    

