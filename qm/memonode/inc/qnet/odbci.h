#define SQL_NULL_DATA             (-1)
#define SQL_DATA_AT_EXEC          (-2)
#define SQL_SUCCESS                0
#define SQL_SUCCESS_WITH_INFO      1

#define SQL_NTS                   (-3)
#define SQL_NTSL                  (-3L)

#define SQL_API

#define SQL_UNKNOWN_TYPE        0
#define SQL_CHAR            1
#define SQL_NUMERIC         2
#define SQL_DECIMAL         3
#define SQL_INTEGER         4
#define SQL_SMALLINT        5
#define SQL_FLOAT           6
#define SQL_REAL            7
#define SQL_DOUBLE          8
#define SQL_VARCHAR        12

#define SQL_COMMIT          0
#define SQL_ROLLBACK        1

typedef unsigned char   SQLCHAR;

typedef long            SQLINTEGER;
typedef unsigned long   SQLUINTEGER;
#define SQLLEN          SQLINTEGER
#define SQLULEN         SQLUINTEGER
#define SQLSETPOSIROW   SQLUSMALLINT
typedef SQLULEN         SQLROWCOUNT;
typedef SQLULEN         SQLROWSETSIZE;
typedef SQLULEN         SQLTRANSID;
typedef SQLLEN          SQLROWOFFSET;

typedef signed short int   SQLSMALLINT;
typedef unsigned short  SQLUSMALLINT;

typedef SQLSMALLINT     SQLRETURN;

typedef void *          SQLPOINTER;
typedef void*           SQLHWND;

#define SQL_CLOSE           0
#define SQL_DROP            1
#define SQL_UNBIND          2
#define SQL_RESET_PARAMS    3

#define SQL_DRIVER_NOPROMPT             0
#define SQL_DRIVER_COMPLETE             1
#define SQL_DRIVER_PROMPT               2
#define SQL_DRIVER_COMPLETE_REQUIRED    3

#define SQL_AUTOCOMMIT_OFF              0UL
#define SQL_AUTOCOMMIT_ON               1UL
#define SQL_AUTOCOMMIT_DEFAULT          SQL_AUTOCOMMIT_ON

#define SQL_C_CHAR    SQL_CHAR             /* CHAR, VARCHAR, DECIMAL, NUMERIC */
#define SQL_C_LONG    SQL_INTEGER          /* INTEGER                      */
#define SQL_C_SHORT   SQL_SMALLINT         /* SMALLINT                     */
#define SQL_C_FLOAT   SQL_REAL             /* REAL                         */
#define SQL_C_DOUBLE  SQL_DOUBLE           /* FLOAT, DOUBLE                */
#define SQL_C_DEFAULT 99

#include "misc/xf.h"
#include "qnet/qapi.h"

SQLRETURN  SQL_API SQLError(SQLHENV EnvironmentHandle,
                            SQLHDBC ConnectionHandle, SQLHSTMT StatementHandle,
                            SQLCHAR *Sqlstate, SQLINTEGER *NativeError,
                            SQLCHAR *MessageText, SQLSMALLINT BufferLength,
                            SQLSMALLINT *TextLength);
SQLRETURN  SQL_API SQLRowCount(SQLHSTMT StatementHandle, SQLLEN *RowCount);
SQLRETURN  SQL_API SQLAllocStmt(SQLHDBC ConnectionHandle, SQLHSTMT *StatementHandle);
SQLRETURN  SQL_API SQLFreeStmt(SQLHSTMT StatementHandle, SQLUSMALLINT Option);
SQLRETURN  SQL_API SQLTransact(SQLHENV EnvironmentHandle,
                               SQLHDBC ConnectionHandle,
                               SQLUSMALLINT CompletionType);
SQLRETURN  SQL_API SQLSetConnectOption(SQLHDBC ConnectionHandle, SQLUSMALLINT Option, SQLULEN Value);
SQLRETURN SQL_API SQLDriverConnect(SQLHDBC            hdbc,
                                   SQLHWND            hwnd,
                                   SQLCHAR           *szConnStrIn,
                                   SQLSMALLINT        cbConnStrIn,
                                   SQLCHAR           *szConnStrOut,
                                   SQLSMALLINT        cbConnStrOutMax,
                                   SQLSMALLINT       *pcbConnStrOut,
                                   SQLUSMALLINT       fDriverCompletion);
SQLRETURN  SQL_API SQLDisconnect(SQLHDBC ConnectionHandle);
SQLRETURN  SQL_API SQLAllocConnect(SQLHENV EnvironmentHandle, SQLHDBC *ConnectionHandle);
SQLRETURN  SQL_API SQLFreeConnect(SQLHDBC ConnectionHandle);
SQLRETURN  SQL_API SQLAllocEnv(SQLHENV *EnvironmentHandle);
SQLRETURN  SQL_API SQLFreeEnv(SQLHENV EnvironmentHandle);
SQLRETURN  SQL_API SQLNumResultCols(SQLHSTMT StatementHandle, SQLSMALLINT *ColumnCount);
SQLRETURN  SQL_API SQLBindCol(SQLHSTMT StatementHandle,
                              SQLUSMALLINT ColumnNumber, SQLSMALLINT TargetType,
                              SQLPOINTER TargetValue, SQLLEN BufferLength,
                              SQLLEN *StrLen_or_Ind);
SQLRETURN  SQL_API SQLDescribeCol(SQLHSTMT StatementHandle,
                                  SQLUSMALLINT ColumnNumber, SQLCHAR *ColumnName,
                                  SQLSMALLINT BufferLength, SQLSMALLINT *NameLength,
                                  SQLSMALLINT *DataType, SQLULEN *ColumnSize,
                                  SQLSMALLINT *DecimalDigits, SQLSMALLINT *Nullable);


