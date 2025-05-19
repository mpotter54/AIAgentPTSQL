from langchain_core.runnables import RunnableWithFallbacks, RunnableLambda
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import AnyMessage, AIMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing import TypedDict
from SQLServerTools import list_tables_tool, create_sql_schema_extractor
from SQLServerTools import get_schema_tool
from SQLServerTools import db_query_tool, db_connection
from typing import Any, Annotated, Literal
import os
import warnings
import gradio as gr

warnings.filterwarnings("ignore", message=".*TqdmWarning.*")

os.environ["GOOGLE_API_KEY"] = "add google api key"


# os.environ["TAVILY_API_KEY"] = "add tavily api key"

# Define the state for the agent
class State(TypedDict):
    task: str
    server: str
    user: str
    password: str
    database: str
    messages: Annotated[list[AnyMessage], add_messages]


class ewriter():
    def __init__(self, memory, server, user, password, database):
        self.server = server
        self.user = user
        self.password = password
        self.database = database
        self.query_check_system = """You are a SQL expert with a strong attention to detail.
        Double check the SQL query for common mistakes, including:
        - Using NOT IN with NULL values
        - Using UNION when UNION ALL should have been used
        - Using BETWEEN for exclusive ranges
        - Data type mismatch in predicates
        - Properly quoting identifiers
        - Using the correct number of arguments for functions
        - Casting to the correct data type
        - Using the proper columns for joins
    
        If there are any of the above mistakes, rewrite the query. If there are no mistakes, just reproduce the original query.
    
        You will call the appropriate tool to execute the query after running this check."""

        self.query_check_prompt = ChatPromptTemplate.from_messages(
            [("system", self.query_check_system), ("placeholder", "{messages}")]
        )
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                          temperature=0,
                                          max_tokens=2048,
                                          timeout=None,
                                          max_retries=2,
                                          max_output_tokens=2048)
        self.llm_with_tools = self.llm.bind_tools([db_query_tool], tool_choice="db_query_tool")
        self.llm_with_query_check = (self.query_check_prompt | self.llm_with_tools);

        # Define a new graph
        self.workflow = StateGraph(State)

        self.workflow.add_node("first_tool_call", self.first_tool_call)

        # Add nodes for the first two tools
        self.workflow.add_node(
            "list_tables_tool", self.create_tool_node_with_fallback([list_tables_tool])
        )
        self.workflow.add_node("get_schema_tool", self.create_tool_node_with_fallback([get_schema_tool]))

        # Add a node for a model to choose the relevant tables based on the question and available tables
        self.llm_schema = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                                 temperature=0,
                                                 max_tokens=2048,
                                                 timeout=None,
                                                 max_retries=2,
                                                 max_output_tokens=2048)
        self.model_get_schema = self.llm_schema.bind_tools([get_schema_tool])
        self.workflow.add_node(
            "model_get_schema",
            lambda state: {
                "messages": [self.model_get_schema.invoke(state["messages"])],
            },
        )

        # Add a node for a model to generate a query based on the question and schema
        self.query_gen_system = """You are a SQL expert with a strong attention to detail.
    
        Given an input question, output a syntactically correct Microsoft SQL Server query to run, then look at the results of the query and return the answer.
    
        DO NOT call any tool besides SubmitFinalAnswer to submit the final answer.
    
        When generating the query:
    
        Output the SQL query that answers the input question without a tool call.
    
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    
        If you get an error while executing a query, rewrite the query and try again.
    
        If you get an empty result set, you should try to rewrite the query to get a non-empty result set. 
        NEVER make stuff up if you don't have enough information to answer the query... just say you don't have enough information.
    
        If you have enough information to answer the input question, simply invoke the appropriate tool to submit the final answer to the user.
    
        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database."""
        self.query_gen_prompt = ChatPromptTemplate.from_messages(
            [("system", self.query_gen_system), ("placeholder", "{messages}")]
        )
        self.llm_qry = ChatGoogleGenerativeAI(model="gemini-2.0-flash",
                                              temperature=0,
                                              max_tokens=2048,
                                              timeout=None,
                                              max_retries=2,
                                              max_output_tokens=2048)
        self.llm_qry_with_tools = self.llm_qry.bind_tools([self.SubmitFinalAnswer])
        self.query_gen = (self.query_gen_prompt | self.llm_qry_with_tools);
        self.workflow.add_node("query_gen", self.query_gen_node)

        # Add a node for the model to check the query before executing it
        self.workflow.add_node("correct_query", self.model_check_query)

        # Add node for executing the query
        self.workflow.add_node("execute_query", self.create_tool_node_with_fallback([db_query_tool]))
        # Specify the edges between the nodes
        self.workflow.add_edge(START, "first_tool_call")
        self.workflow.add_edge("first_tool_call", "list_tables_tool")
        self.workflow.add_edge("list_tables_tool", "model_get_schema")
        self.workflow.add_edge("model_get_schema", "get_schema_tool")
        self.workflow.add_edge("get_schema_tool", "query_gen")
        self.workflow.add_conditional_edges(
            "query_gen",
            self.should_continue,
        )
        self.workflow.add_edge("correct_query", "execute_query")
        self.workflow.add_edge("execute_query", "query_gen")
        self.memory = memory
        self.graph = self.workflow.compile(checkpointer=self.memory)
        db_connection.set_db_connection(server, user, password, database)

    def create_tool_node_with_fallback(self, tools: list) -> RunnableWithFallbacks[Any, dict]:
        """
        Create a ToolNode with a fallback to handle errors and surface them to the agent.
        """
        return ToolNode(tools).with_fallbacks(
            [RunnableLambda(self.handle_tool_error)], exception_key="error"
        )

    def handle_tool_error(self, state) -> dict:
        error = state.get("error")
        tool_calls = state["messages"][-1].tool_calls
        return {
            "messages": [
                ToolMessage(
                    content=f"Error: {repr(error)}\n please fix your mistakes.",
                    tool_call_id=tc["id"],
                )
                for tc in tool_calls
            ]
        }

    # Add a node for the first tool call
    def first_tool_call(self, state: State) -> dict[str, list[AIMessage]]:
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "list_tables_tool",
                            "args": {},
                            "id": "tool_abcd123",
                        }
                    ],
                )
            ]
        }

    def model_check_query(self, state: State) -> dict[str, list[AIMessage]]:
        """
        Use this tool to double-check if your query is correct before executing it.
        """
        return {"messages": [self.llm_with_query_check.invoke({"messages": [state["messages"][-1]]})]}

    # Describe a tool to represent the end state
    class SubmitFinalAnswer(BaseModel):
        """Submit the final answer to the user based on the query results."""

        final_answer: str = Field(..., description="The final answer to the user")

    def query_gen_node(self, state: State):
        message = self.query_gen.invoke(state)

        # Sometimes, the LLM will hallucinate and call the wrong tool.
        # We need to catch this and return an error message.
        tool_messages = []
        if message.tool_calls:
            for tc in message.tool_calls:
                if tc["name"] != "SubmitFinalAnswer":
                    tool_messages.append(
                        ToolMessage(
                            content=f"Error: The wrong tool was called: {tc['name']}. Please fix your mistakes. Remember to only call SubmitFinalAnswer to submit the final answer. Generated queries should be outputted WITHOUT a tool call.",
                            tool_call_id=tc["id"],
                        )
                    )
        else:
            tool_messages = []
        return {"messages": [message] + tool_messages}

    # Define a conditional edge to decide whether to continue or end the workflow
    def should_continue(self, state: State) -> Literal[END, "correct_query", "query_gen"]:
        messages = state["messages"]
        last_message = messages[-1]
        # If there is a tool call, then we finish
        if getattr(last_message, "tool_calls", None):
            return END
        if last_message.content.startswith("Error:") or len(last_message.content) < 1:
            return "query_gen"
        else:
            return "correct_query"


class writer_gui():
    def __init__(self, graph, server, user, password, database, share=False):
        self.graph = graph
        self.share = share
        self.partial_message = ""
        self.response = {}
        self.max_iterations = 10
        self.iterations = []
        self.threads = []
        self.thread_id = -1
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        self.server = server
        self.user = user
        self.password = password
        self.database = database
        self.demo = self.create_interface()

    def run_agent(self, start, topic, stop_after, sql_server, sql_database, sql_user_name, sql_password):
        if start:
            self.iterations.append(0)
            self.thread_id += 1  # new agent, new thread
            self.threads.append(self.thread_id)
            config = None
        self.thread = {"configurable": {"thread_id": str(self.thread_id)}}
        self.server = sql_server
        self.database = sql_database
        self.user = sql_user_name
        self.password = sql_password
        db_connection.set_db_connection(self.server, self.user, self.password, self.database)
        create_sql_schema_extractor()
        sql_qry = {'task': topic,
                   'server': self.server,
                   'user': self.user,
                   'password': self.password,
                   'database': self.database,
                   "messages": [("user", topic)]}
        self.response = self.graph.invoke(sql_qry, self.thread)
        # json_str = self.response["messages"][-1].tool_calls[0]["args"]["final_answer"]
        self.iterations[self.thread_id] += 1
        self.partial_message += str(self.response)
        self.partial_message += f"\n------------------\n\n"
        # print("Hit the end")
        return

    def get_disp_state(self, ):
        current_state = self.graph.get_state(self.thread)
        # lnode = current_state.values["lnode"]
        # acount = current_state.values["count"]
        # rev = current_state.values["revision_number"]
        nnode = current_state.next
        #print  (lnode,nnode,self.thread_id,rev,acount)
        # return lnode, nnode, self.thread_id, rev, acount
        return nnode, self.thread_id

    def get_state(self, key):
        current_values = self.graph.get_state(self.thread)
        if key in current_values.values:
            nnode, self.thread_id, rev, astep = self.get_disp_state()
            new_label = f"thread_id: {self.thread_id}, step: {astep}"
            return gr.update(label=new_label, value=current_values.values[key])
        else:
            return ""

    def get_content(self, ):
        current_values = self.graph.get_state(self.thread)
        if "content" in current_values.values:
            content = current_values.values["content"]
            nnode, thread_id, astep = self.get_disp_state()
            new_label = f"thread_id: {self.thread_id}, step: {astep}"
            return gr.update(label=new_label, value="\n\n".join(item for item in content) + "\n\n")
        else:
            return ""

    def update_hist_pd(self, ):
        #print("update_hist_pd")
        hist = []
        # curiously, this generator returns the latest first
        for state in self.graph.get_state_history(self.thread):
            if state.metadata['step'] < 1:
                continue
            thread_ts = state.config['configurable']['thread_ts']
            tid = state.config['configurable']['thread_id']
            # count = state.values['count']
            # lnode = state.values['lnode']
            # rev = state.values['revision_number']
            nnode = state.next
            st = f"{tid}:{nnode}:{thread_ts}"
            hist.append(st)
        return gr.Dropdown(label="update_state from: thread:last_node:next_node:rev:thread_ts",
                           choices=hist, value=hist[0], interactive=True)

    def find_config(self, thread_ts):
        for state in self.graph.get_state_history(self.thread):
            config = state.config
            if config['configurable']['thread_ts'] == thread_ts:
                return config
        return (None)

    def copy_state(self, hist_str):
        ''' result of selecting an old state from the step pulldown. Note does not change thread.
             This copies an old state to a new current state.
        '''
        thread_ts = hist_str.split(":")[-1]
        # print(f"copy_state from {thread_ts}")
        config = self.find_config(thread_ts)
        # print(config)
        state = self.graph.get_state(config)
        self.graph.update_state(self.thread, state.values, as_node=state.values['lnode'])
        new_state = self.graph.get_state(self.thread)  # should now match
        new_thread_ts = new_state.config['configurable']['thread_ts']
        tid = new_state.config['configurable']['thread_id']
        # count = new_state.values['count']
        # lnode = new_state.values['lnode']
        # rev = new_state.values['revision_number']
        nnode = new_state.next
        return nnode, new_thread_ts

    def update_thread_pd(self, ):
        # print("update_thread_pd")
        return gr.Dropdown(label="choose thread", choices=self.threads, value=self.thread_id, interactive=True)

    def switch_thread(self, new_thread_id):
        # print(f"switch_thread{new_thread_id}")
        self.thread = {"configurable": {"thread_id": str(new_thread_id)}}
        self.thread_id = new_thread_id
        return

    def modify_state(self, key, asnode, new_state):
        ''' gets the current state, modifes a single value in the state identified by key, and updates state with it.
        note that this will create a new 'current state' node. If you do this multiple times with different keys, it will create
        one for each update. Note also that it doesn't resume after the update
        '''
        current_values = self.graph.get_state(self.thread)
        current_values.values[key] = new_state
        self.graph.update_state(self.thread, current_values.values, as_node=asnode)
        return

    def create_interface(self):
        #         def_qry = '''Get all the Sales.Store where SalesPersonId equals 290.
        # Please include Sales.Store.Name, SalesPersonId, and Person.LastName as SalesLastName in the final answer.
        # Please use a grid format also.
        #                    '''
        def_qry = '''Get the top 100 Sales.SalesPerson.BusinessEntityID with the most Sales.SalesPerson.SalesLastYear 
        in grid format. Please include Sales.SalesPerson.BusinessEntityID, Sales.SalesPerson.SalesLastYear, 
        Person.LastName,and Person.FirstName in the final answer. Please order by Sales.SalesPerson.SalesLastYear in 
        descending order. Please format SalesLastYear in currency format rounded to the whole dollar.  Show zero 
        decimal places. Please only show rows where SalesLastYear is greater than zero.'''

        with gr.Blocks(theme=gr.themes.Default(spacing_size='sm', text_size="sm")) as demo:

            def updt_disp():
                ''' general update display on state change '''
                json_str: str = '*****\n'
                current_state = self.graph.get_state(self.thread)
                hist = []
                # curiously, this generator returns the latest first
                for state in self.graph.get_state_history(self.thread):
                    if state.metadata['step'] < 1:  # ignore early states
                        continue
                    if "thread_ts" in state.config:
                        s_thread_ts = state.config['configurable']['thread_ts']
                    else:
                        s_thread_ts = ''
                    s_tid = state.config['configurable']['thread_id']
                    # s_count = state.values['count']
                    # s_lnode = state.values['lnode']
                    # s_rev = state.values['revision_number']
                    s_nnode = state.next
                    st = f"{s_tid}:{s_nnode}:{s_thread_ts}"
                    hist.append(st)
                if not current_state.metadata:  # handle init call
                    return {}
                else:
                    if len(self.response) < 1:
                        for msg in current_state[0]['messages']:
                            if len(msg.content) > 0 and isinstance(msg, AIMessage):
                                json_str += msg.content
                                json_str += '\n'
                        json_str += '*****'
                    else:
                        json_str = self.response["messages"][-1].tool_calls[0]["args"]["final_answer"]
                    return {
                        topic_bx: current_state.values["task"],
                        sql_server_bx: self.server,
                        sql_database_bx: self.database,
                        sql_user_name_bx: self.user,
                        sql_password_bx: self.password,
                        threadid_bx: self.thread_id,
                        live: json_str,
                        thread_pd: gr.Dropdown(label="choose thread", choices=self.threads, value=self.thread_id,
                                               interactive=True),
                        step_pd: gr.Dropdown(label="update_state from: thread:count:last_node:next_node:rev:thread_ts",
                                             choices=hist, value=hist[0], interactive=True),
                    }

            def get_snapshots():
                new_label = f"thread_id: {self.thread_id}, Summary of snapshots"
                sstate = ""
                for state in self.graph.get_state_history(self.thread):
                    for key in ['plan', 'draft', 'critique']:
                        if key in state.values:
                            state.values[key] = state.values[key][:80] + "..."
                    if 'content' in state.values:
                        for i in range(len(state.values['content'])):
                            state.values['content'][i] = state.values['content'][i][:20] + '...'
                    if 'writes' in state.metadata:
                        state.metadata['writes'] = "not shown"
                    sstate += str(state) + "\n\n"
                return gr.update(label=new_label, value=sstate)

            def vary_btn(stat):
                # print(f"vary_btn{stat}")
                return gr.update(variant=stat)

            with gr.Tab("SQL Query"):
                with gr.Row():
                    topic_bx = gr.Textbox(label="Query", value=def_qry, lines=10, max_lines=10)
                    gen_btn = gr.Button("Run Query", scale=0, min_width=80, variant='primary')
                    cont_btn = gr.Button("Continue Execution", scale=0, min_width=80, visible=False)
                with gr.Row():
                    sql_server_bx = gr.Textbox(label="Server", min_width=100, value=self.server, interactive=True)
                    sql_database_bx = gr.Textbox(label="Database", min_width=100, value=self.database, interactive=True)
                    sql_user_name_bx = gr.Textbox(label="User", scale=0, min_width=80, value=self.user,
                                                  interactive=True)
                    sql_password_bx = gr.Textbox(label="Password", scale=0, min_width=80, type="password",
                                                 value=self.password, interactive=True)
                    threadid_bx = gr.Textbox(label="Thread", scale=0, min_width=10, visible=False)
                with gr.Accordion("Manage Agent", open=False):
                    checks = list(self.graph.nodes.keys())
                    checks.remove('__start__')
                    stop_after = gr.CheckboxGroup(checks, label="Interrupt After State", value=checks, scale=0,
                                                  min_width=400, visible=False)
                    with gr.Row():
                        thread_pd = gr.Dropdown(choices=self.threads, interactive=True, label="select thread",
                                                min_width=120, scale=0)
                        step_pd = gr.Dropdown(choices=['N/A'], interactive=True, label="select step", min_width=160,
                                              scale=1)

                live = gr.Textbox(label="Query Output", lines=25, max_lines=25)

                # actions
                sdisps = [topic_bx, sql_server_bx, sql_database_bx, sql_user_name_bx, sql_password_bx, step_pd,
                          threadid_bx, thread_pd, live]
                thread_pd.input(self.switch_thread, [thread_pd], None).then(
                    fn=updt_disp, inputs=None, outputs=sdisps)
                step_pd.input(self.copy_state, [step_pd], None).then(
                    fn=updt_disp, inputs=None, outputs=sdisps)
                gen_btn.click(vary_btn, gr.Number("secondary", visible=False), gen_btn).then(
                    fn=self.run_agent, inputs=[gr.Number(True, visible=False),
                                               topic_bx,
                                               stop_after,
                                               sql_server_bx,
                                               sql_database_bx,
                                               sql_user_name_bx,
                                               sql_password_bx], outputs=[live],
                    show_progress=True).then(
                    fn=updt_disp, inputs=None, outputs=sdisps).then(
                    vary_btn, gr.Number("primary", visible=False), gen_btn).then(
                    vary_btn, gr.Number("primary", visible=False), cont_btn)
                cont_btn.click(vary_btn, gr.Number("secondary", visible=False), cont_btn).then(
                    fn=self.run_agent, inputs=[gr.Number(False, visible=False),
                                               topic_bx,
                                               stop_after,
                                               sql_server_bx,
                                               sql_database_bx,
                                               sql_user_name_bx,
                                               sql_password_bx],
                    outputs=[live]).then(
                    fn=updt_disp, inputs=None, outputs=sdisps).then(
                    vary_btn, gr.Number("primary", visible=False), cont_btn)
            with gr.Tab("StateSnapShots"):
                with gr.Row():
                    refresh_btn = gr.Button("Refresh")
                snapshots = gr.Textbox(label="State Snapshots Summaries")
                refresh_btn.click(fn=get_snapshots, inputs=None, outputs=snapshots)
        return demo

    def launch(self, share=None):
        if port := os.getenv("PORT1"):
            self.demo.launch(share=True, server_port=int(port), server_name="0.0.0.0")
        else:
            self.demo.launch(share=self.share)
