import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from SQLServerAgent import ewriter, writer_gui

server = "add DB server name for MSSQL 2019"
user = "add DB user for MSSQL AdventureWorks"
password = "add DB user password for MSSQL AdventureWorks"
database = "add DB name for MSSQL AdventureWorks restore AdventureWorks2019.bak"

memory = SqliteSaver(conn=sqlite3.connect(":memory:", check_same_thread=False))
MultiAgent = ewriter(memory, server, user, password, database)


# messages = MultiAgent.graph.invoke(
#     {"messages": [("user", "Get all the Sales.Store where SalesPersonId equals 290. "
#                    "Please include Sales.Store.Name, SalesPersonId, and Person.LastName as SalesLastName "
#                    "in the final answer.  Please use a grid format also.")
#                   ]}
# )
# json_str = messages["messages"][-1].tool_calls[0]["args"]["final_answer"]
# print(json_str)
#
# messages = MultiAgent.graph.invoke(
#     {"messages": [("user", "Get the top 3 Sales.SalesPerson.BusinessEntityID with the most "
#                    "Sales.SalesPerson.SalesLastYear in grid format. Please include "
#                    "Sales.SalesPerson.BusinessEntityID, Sales.SalesPerson.SalesLastYear, Person.LastName,"
#                    "and Person.FirstName in the final answer. ")
#                   ]}
# )
# json_str = messages["messages"][-1].tool_calls[0]["args"]["final_answer"]
# print(json_str)

# Get the top 100 Sales.SalesPerson.BusinessEntityID with the most Sales.SalesPerson.SalesLastYear in grid format.
# Please include Sales.SalesPerson.BusinessEntityID, Sales.SalesPerson.SalesLastYear, Person.LastName,and Person.FirstName in the final answer.
# Please order by Sales.SalesPerson.SalesLastYear in descending order.
# Please format SalesLastYear in currency format rounded to the whole dollar.  Show zero decimal places.
# Please only show rows where SalesLastYear is greater than zero.

app = writer_gui(MultiAgent.graph, server, user, password, database)
app.launch()
