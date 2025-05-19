import pymssql
from pydantic import BaseModel
from pymssql import _mssql
from langchain_core.tools import tool

test_table = "Sales.SalesPerson"
test_qry = ("SELECT sp.[BusinessEntityID]"
            ",e.NationalIDNumber"
            ",sp.[SalesYTD]"
            ",sp.[SalesLastYear]"
            "FROM [Sales].[SalesPerson] sp inner join "
            "[HumanResources].[Employee] e on sp.BusinessEntityID = e.BusinessEntityID")


class DBConnection:
    db_server: str
    db_user: str
    db_password: str
    db_database: str

    def set_db_connection(self, server: str, user: str, password: str, database: str):
        self.db_server = server
        self.db_user = user
        self.db_password = password
        self.db_database = database


db_connection = DBConnection()


def create_sql_schema_extractor():
    proc = """CREATE OR ALTER PROC usp_GenerateCreateTable
    @table_name SYSNAME
AS
    DECLARE @object_name SYSNAME,@object_id INT
    DECLARE @SQL nvarchar(max) = ''
    DECLARE @SAMPLE_QRY nvarchar(max) = 'Select top 3 * from ' + @TABLE_NAME + ' order by 1'

    SELECT @object_name = '"' + o.NAME + '"',@object_id = o.[object_id] --'"' + s.NAME + '"."' + o.NAME + '"',@object_id = o.[object_id]
    FROM sys.objects o WITH (NOWAIT)
    JOIN sys.schemas s WITH (NOWAIT) ON o.[schema_id] = s.[schema_id]
    WHERE s.NAME + '.' + o.NAME = @table_name
        AND o.[type] = 'U'
        AND o.is_ms_shipped = 0

    set @SQL = ''
    ;WITH index_column
    AS (
        SELECT ic.[object_id]
            ,ic.index_id
            ,ic.is_descending_key
            ,ic.is_included_column
            ,c.NAME
        FROM sys.index_columns ic WITH (NOWAIT)
        JOIN sys.columns c WITH (NOWAIT) ON ic.[object_id] = c.[object_id]
            AND ic.column_id = c.column_id
        WHERE ic.[object_id] = @object_id
        )
        ,fk_columns
    AS (
        SELECT k.constraint_object_id
            ,cname = c.NAME
            ,rcname = rc.NAME
        FROM sys.foreign_key_columns k WITH (NOWAIT)
        JOIN sys.columns rc WITH (NOWAIT) ON rc.[object_id] = k.referenced_object_id
            AND rc.column_id = k.referenced_column_id
        JOIN sys.columns c WITH (NOWAIT) ON c.[object_id] = k.parent_object_id
            AND c.column_id = k.parent_column_id
        WHERE k.parent_object_id = @object_id
        )
    SELECT @SQL = 'CREATE TABLE ' + @object_name + CHAR(10) + '(' + CHAR(10) + STUFF((
                SELECT CHAR(9) + ', "' + c.NAME + '" ' + CASE
                        WHEN c.is_computed = 1
                            THEN 'AS ' + cc.[definition]
                        ELSE UPPER(tp.NAME) + CASE
                                WHEN tp.NAME IN (
                                        'varchar'
                                        ,'char'
                                        ,'varbinary'
                                        ,'binary'
                                        ,'text'
                                        )
                                    THEN '(' + CASE
                                            WHEN c.max_length = - 1
                                                THEN 'MAX'
                                            ELSE CAST(c.max_length AS VARCHAR(5))
                                            END + ')'
                                WHEN tp.NAME IN (
                                        'nvarchar'
                                        ,'nchar'
                                        ,'ntext'
                                        )
                                    THEN '(' + CASE
                                            WHEN c.max_length = - 1
                                                THEN 'MAX'
                                            ELSE CAST(c.max_length / 2 AS VARCHAR(5))
                                            END + ')'
                                WHEN tp.NAME IN (
                                        'datetime2'
                                        ,'time2'
                                        ,'datetimeoffset'
                                        )
                                    THEN '(' + CAST(c.scale AS VARCHAR(5)) + ')'
                                WHEN tp.NAME = 'decimal'
                                    THEN '(' + CAST(c.[precision] AS VARCHAR(5)) + ',' + CAST(c.scale AS VARCHAR(5)) + ')'
                                ELSE ''
                                END + CASE
                                WHEN c.collation_name IS NOT NULL
                                    THEN ' COLLATE ' + c.collation_name
                                ELSE ''
                                END + CASE
                                WHEN c.is_nullable = 1
                                    THEN ' NULL'
                                ELSE ' NOT NULL'
                                END + CASE
                                WHEN dc.[definition] IS NOT NULL
                                    THEN ' DEFAULT' + dc.[definition]
                                ELSE ''
                                END + CASE
                                WHEN ic.is_identity = 1
                                    THEN ' IDENTITY(' + CAST(ISNULL(ic.seed_value, '0') AS CHAR(1)) + ',' + CAST(ISNULL(ic.increment_value, '1') AS CHAR(1)) + ')'
                                ELSE ''
                                END
                        END + CHAR(10)
                FROM sys.columns c WITH (NOWAIT)
                JOIN sys.types tp WITH (NOWAIT) ON c.user_type_id = tp.user_type_id
                LEFT JOIN sys.computed_columns cc WITH (NOWAIT) ON c.[object_id] = cc.[object_id]
                    AND c.column_id = cc.column_id
                LEFT JOIN sys.default_constraints dc WITH (NOWAIT) ON c.default_object_id != 0
                    AND c.[object_id] = dc.parent_object_id
                    AND c.column_id = dc.parent_column_id
                LEFT JOIN sys.identity_columns ic WITH (NOWAIT) ON c.is_identity = 1
                    AND c.[object_id] = ic.[object_id]
                    AND c.column_id = ic.column_id
                WHERE c.[object_id] = @object_id
                ORDER BY c.column_id
                FOR XML PATH('')
                    ,TYPE
                ).value('.', 'NVARCHAR(MAX)'), 1, 2, CHAR(9) + ' ') + ISNULL((
                SELECT CHAR(9) + ', PRIMARY KEY (' + (
                        SELECT STUFF((
                                    SELECT ', "' + c.NAME + '" ' + CASE
                                            WHEN ic.is_descending_key = 1
                                                THEN 'DESC'
                                            ELSE 'ASC'
                                            END
                                    FROM sys.index_columns ic WITH (NOWAIT)
                                    JOIN sys.columns c WITH (NOWAIT) ON c.[object_id] = ic.[object_id]
                                        AND c.column_id = ic.column_id
                                    WHERE ic.is_included_column = 0
                                        AND ic.[object_id] = k.parent_object_id
                                        AND ic.index_id = k.unique_index_id
                                    FOR XML PATH(N'')
                                        ,TYPE
                                    ).value('.', 'NVARCHAR(MAX)'), 1, 2, '')
                        ) + ')' + CHAR(10)
                FROM sys.key_constraints k WITH (NOWAIT)
                WHERE k.parent_object_id = @object_id
                    AND k.[type] = 'PK'
                ), '') +
                ISNULL(
                    (
                        STUFF((
                        SELECT CHAR(9) + ', FOREIGN KEY(' + STUFF((
                                    SELECT ', "' + k.cname + '"'
                                    FROM fk_columns k
                                    WHERE k.constraint_object_id = fk.[object_id]
                                    FOR XML PATH('')
                                        ,TYPE
                                    ).value('.', 'NVARCHAR(MAX)'), 1, 2, '') + ')' + ' REFERENCES "' + ro.NAME + '" (' + STUFF((
                                    SELECT ', "' + k.rcname + '"'
                                    FROM fk_columns k
                                    WHERE k.constraint_object_id = fk.[object_id]
                                    FOR XML PATH('')
                                        ,TYPE
                                    ).value('.', 'NVARCHAR(MAX)'), 1, 2, '') + ')' + CHAR(10)
                        FROM sys.foreign_keys fk WITH (NOWAIT)
                        JOIN sys.objects ro WITH (NOWAIT) ON ro.[object_id] = fk.referenced_object_id
                        WHERE fk.parent_object_id = @object_id
                        FOR XML PATH(N'')
                            ,TYPE
                        ).value('.', 'NVARCHAR(MAX)'), 1, 2, CHAR(9) + ',')
                ), '') + ')'
                select @SQL as "TBL_SCHEMA"
                EXECUTE sp_executesql @SAMPLE_QRY
"""
    try:
        conn = _mssql.connect(server=db_connection.db_server,
                              user=db_connection.db_user,
                              password=db_connection.db_password,
                              database=db_connection.db_database)
        conn.execute_non_query(proc)
    except pymssql.Error as e:
        print(f"Error: {e}")


# create_sql_schema_extractor()


@tool
def list_tables_tool() -> str:
    """
    Retrieves a list of all table names from a SQL Server database.

    Args:
        server (str): The SQL Server instance.
        database (str): The database name.
        user (str): The database user.
        password (str): The database password.

    Returns:
        list: A list of table names.
    """

    try:
        conn = _mssql.connect(server=db_connection.db_server,
                              user=db_connection.db_user,
                              password=db_connection.db_password,
                              database=db_connection.db_database)

        # Execute the SQL query to fetch table names
        conn.execute_query("SELECT TABLE_SCHEMA + '.' + TABLE_NAME as TABLE_NAME FROM INFORMATION_SCHEMA.TABLES "
                           "WHERE TABLE_TYPE = 'BASE TABLE' ORDER BY TABLE_NAME")

        # Fetch all rows from the result set
        table_names = ', '.join([row['TABLE_NAME'] for row in conn])

        conn.close()
        return table_names

    except pymssql.Error as e:
        print(f"Error: {e}")
        return ''


@tool
def get_schema_tool(table_name: str) -> str:
    """
    Retrieves the schema for a table.

    Args:
        table_name (str): The SQL Server table name.

    Returns:
        the table_name schema as a str plus 3 sample rows:
        :param table_name:
    """

    try:
        conn = _mssql.connect(server=db_connection.db_server,
                              user=db_connection.db_user,
                              password=db_connection.db_password,
                              database=db_connection.db_database)
        # if table_name.find("dbo.") == -1:
        #     table_name = "dbo." + table_name

        # Execute the SQL query to fetch table schema
        schema_qry = f"exec usp_GenerateCreateTable '{table_name}'"
        conn.execute_query(schema_qry)

        # Fetch table schema from row
        for row in conn:
            table_schema = row['TBL_SCHEMA']
        # Fetch table sample rows
        columns_defined: int = 0
        column_names: str = ''
        sample_rows: str = []
        for row in conn:
            col = 0
            if columns_defined == 0:
                for key, value in row.items():
                    if col % 2 == 0:
                        col += 1
                        continue
                    else:
                        col += 1
                        if len(column_names) == 0:
                            column_names = str(key)
                        else:
                            column_names += f", {key}"
                columns_defined = 1
            col: int = 0
            row_string: str = ''
            for key, value in row.items():
                if col % 2 == 0:
                    col += 1
                    continue
                else:
                    col += 1
                    if len(row_string) == 0:
                        row_string = str(value)
                    else:
                        row_string += f", {str(value)}"
            sample_rows.append(row_string)
        conn.close()
        table_schema_plus_sample_rows = table_schema
        table_schema_plus_sample_rows += "\n\n"
        table_schema_plus_sample_rows += (column_names + "\n")
        for r in sample_rows:
            table_schema_plus_sample_rows += (r + "\n")
        return table_schema_plus_sample_rows

    except pymssql.Error as e:
        print(f"Error: {e}")
        return ''


@tool
def db_query_tool(query: str) -> str:
    """
    Retrieves the results for a sql query.

    Args:
        query (str): The SQL Server query string.

    Returns:
        the query results as a str:
        :param query:
    """

    try:
        conn = _mssql.connect(server=db_connection.db_server,
                              user=db_connection.db_user,
                              password=db_connection.db_password,
                              database=db_connection.db_database)
        conn.execute_query(query)

        columns_defined: int = 0
        column_names: str = ''
        query_rows = []
        for row in conn:
            col: int = 0
            if columns_defined == 0:
                for key, value in row.items():
                    col += 1
                    if len(column_names) == 0:
                        column_names = str(key)
                    else:
                        column_names += f", {key}"
                columns_defined = 1
            col = 0
            row_string: str = []
            for key, value in row.items():
                col += 1
                row_string.append(value)
            query_rows.append(row_string)
        conn.close()
        return query_rows

    except pymssql.Error as e:
        print(f"Error: {e}")
        query_rows = []
        return query_rows


tools = [
    list_tables_tool,
    get_schema_tool,
    db_query_tool,
]

# print(list_tables_tool.invoke(''))
# print(get_schema_tool.invoke(test_table))
# print(db_query_tool.invoke(test_qry))
