import mysql.connector
import os  # Checking file existence
import pandas as pd  # Reading CSV files
import pymysql
from sqlalchemy import create_engine
import random


### Database credentials for MySQL ###
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'MC1007unicornlay'
DB_NAME = 'chatdb'  # MySQL database name



### Connect to MySQL database ###
def connect_to_mysql():
    try:
        connection = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        return connection
    except Exception as e:
        print(f"Couldn't connected to MySQL: {e}")
        return None

def create_sqlalchemy_engine():
    DB_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    return create_engine(DB_URI)



### Check, upload and process a CSV file to MySQL ###
def upload_csv_to_mysql(file_path, table_name):
    try:
        # Connect to MySQL
        connection = connect_to_mysql()
        if not connection:
            return

        # Read the CSV file into a DataFrame
        data = pd.read_csv(file_path)
        print("CSV file loaded successfully.")

        # Use SQLAlchemy for the database connection
        engine = create_engine(f'mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}')
        
        # Upload the DataFrame to MySQL
        data.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
        print(f"Data from '{file_path}' uploaded to MySQL table '{table_name}'.")
    except Exception as e:
        print(f"Error uploading CSV to MySQL: {e}")
    finally:
        if connection:
            connection.close()

def check_file_then_upload(file_path):
    # Check if the file exists
    if not file_path.endswith(".csv"):
        print("Error: Only .csv files are supported!")
        return None

    # Check if the file is a CSV
    elif not os.path.isfile(file_path):
        print("Error: File does not exist!")
        return None

    # Process and upload the file
    else:
        return process_csv(file_path)

def process_csv(file_path):
    try:
        # Connect to MySQL
        connection = connect_to_mysql()
        if not connection:
            return
        
        # Define a table name for the file that user uploads
        base_name = os.path.splitext(os.path.basename(file_path))[0]  # Get file name without extension
        table_name = base_name.replace(" ", "_").capitalize()  # Replace spaces with underscores, capitalize
        
        # Creating a table in MySQL from user's uploaded file
        create_table_from_dataset(file_path, table_name)

        # Inserting the data into the created table
        insert_data_into_table(file_path, table_name)

        return table_name

    except Exception as e:
        print(f"Error processing CSV: {e}")
        return None



### Create table at MySQL based on file and insert data into table ###
def create_table_from_dataset(file_path, table_name):
    
    # Connect to MySQL
    connection = connect_to_mysql() 
    if connection is None:
        return

    df = pd.read_csv(file_path)
    column_types = infer_sql_column_types(df) # Infer SQL column types

    
    cursor = connection.cursor()
    try:
        # Check if the table already exists
        cursor.execute(f"SHOW TABLES LIKE '{table_name}';")
        result = cursor.fetchone()
        if result:
            print(f"Table '{table_name}' already exists in the database. Skipping table creation.")
            return
        
        # Extract column names from CSV
        csv_columns = df.columns.tolist()

        # Build SQL query dynamically
        create_table_query = f"CREATE TABLE {table_name} ("
        for column in csv_columns:
            sanitized_column = column.replace(" ", "_").replace("-", "_")
            sql_type = column_types.get(column, "TEXT")  # Default to TEXT if type is missing
            create_table_query += f"{sanitized_column} {sql_type}, "
        create_table_query = create_table_query.rstrip(", ") + ")"

        # Execute the CREATE TABLE query
        cursor.execute(create_table_query) 
        print(f"Table '{table_name}' created successfully in MySQL!")
        print("Please hold on while we insert the data for you!")
    
    except Exception as err:
        print(f"Error creating table: {err}")
    
    finally:
        cursor.close()
        connection.close()

def infer_sql_column_types(df):
    column_types = {}
    # Identify unsuitable columns for aggregation
    unsuitable_columns = identify_unsuitable_columns(df)  
    for column in df.columns:
        dtype = df[column].dtype

        # Skip unsuitable columns
        if column in unsuitable_columns:
            continue

        # Map pandas dtype to MySQL type
        if pd.api.types.is_integer_dtype(dtype):
            column_types[column] = "INT"
        elif pd.api.types.is_float_dtype(dtype):
            column_types[column] = "DECIMAL(10,2)"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            column_types[column] = "DATETIME"
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
            column_types[column] = "VARCHAR(255)"
        else:
            column_types[column] = "TEXT"  # Default fallback for unsupported types

    return column_types

def identify_unsuitable_columns(df):
    # Identify columns unsuitable for aggregation based on their name and data characteristics.
    unsuitable_columns = []

    # Heuristic 1: Column names indicating IDs or keys
    identifier_keywords = ["id", "key", "code", "reference"]
    for column in df.columns:
        if any(keyword in column.lower() for keyword in identifier_keywords):
            unsuitable_columns.append(column)

    # Heuristic 2: Mostly unique values (e.g., potential primary keys)
    for column in df.columns:
        if df[column].nunique() / len(df) > 0.9:  # More than 90% unique values
            unsuitable_columns.append(column)

    return set(unsuitable_columns)  # Return as a set to avoid duplicates

def insert_data_into_table(file_path, table_name):
    # Connect to MySQL
    connection = connect_to_mysql() 
    if connection is None:
        return
    
    df = pd.read_csv(file_path)

    # Generating the data
    cursor = connection.cursor()
    try:
        for _, row in df.iterrows(): # Insert rows into the MySQL table
            row_data = tuple(row)  # Convert row to a tuple
            placeholders = ", ".join(["%s"] * len(row))  # Generate placeholders for the query
            columns = ", ".join([col.replace(" ", "_").replace("-", "_") for col in df.columns])
            placeholders = ", ".join(["%s"] * len(df.columns))
            insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            cursor.execute(insert_query, row_data)
        
        connection.commit()  # Commit the transaction
        print(f"Data inserted into table '{table_name}' successfully!")
    except mysql.connector.Error as err:
        print(f"Error inserting data: {err}")
    finally:
        cursor.close()
        connection.close()



### Handle user input and questions
def handle_user_input(file_path,table_name):
    print("\nPlease choose the below options of what would you like to do from here? Type 'exit' to leave ChatDB.")
    print("\n#1. Eplore your database. (See your table , attributes, data, etc.)")
    print("\n#2. Obtain sample queries.")
    print("\n#3. Obtain sample queries with specific language constructs.")
    print("\n#4. Ask specific queries regarding your data file.")
    while True:
        user_input = input("\nEnter the number of your choice: ").strip().lower()

        # Exit condition
        if user_input == "exit":
            print("Exiting ChatDB. Goodbye!")
            break

        # Option #1: Eplore your database
        elif user_input == "1":
            explore_database(table_name)

        # Option #2: Obtain sample queries
        elif user_input == "2":
            generate_queries_dynamically(table_name)

        # Option #3: Obtain sample queries with specific language constructs
        elif user_input == "3":
            handle_specific_construct_option(table_name)
        
        # Option #4: Ask specific queries regarding user's data file
        elif user_input == "4":
            handle_specific_query(table_name)
        
        else:
            print("Options not recognized. Please give me valid options (or type 'exit' to quit.")
        
        print("-" * 50)
        print("\nIf you still like to explore more, please give me the number of the options or type 'exit' to leave ChatDB.")
        print("\n#1. Eplore your database. (See your table , attributes, data, etc.)")
        print("\n#2. Obtain sample queries.")
        print("\n#3. Obtain sample queries with specific language constructs.")
        print("\n#4. Ask specific queries regarding your data file.")



### Function for option #1: Display metadata and prview of the table
def explore_database(table_name):
    # Connect to MySQL
    connection = connect_to_mysql()
    if connection is None:
        print("Failed to connect to the database. Cannot explore the database.")
        return

    cursor = connection.cursor()
    try:
        # Fetch column information
        cursor.execute(f"SHOW COLUMNS FROM {table_name};")
        columns = cursor.fetchall()
        print(f"\nThe table '{table_name}' contains the following attributes:")
        for column in columns:
            print(f" - {column[0]} ({column[1]})")  # Column name and data type

        # Display the first five rows of the table
        print(f"\nHere is a preview of the first 5 rows from the '{table_name}' table:")
        print("-" * 50)
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
        rows = cursor.fetchall()
        if rows:
            # Fetch column names for headers
            cursor.execute(f"SHOW COLUMNS FROM {table_name};")
            column_names = [col[0] for col in cursor.fetchall()]
            print("\t".join(column_names))  # Print column headers
            for row in rows:
                print("\t".join(map(str, row)))  # Print each row
        else:
            print(f"The table '{table_name}' is empty.")
    except Exception as e:
        print(f"Error exploring the database: {e}")
    finally:
        cursor.close()
        connection.close()



### Function for option #2: Generate sample queries
def fetch_table_schema(table_name):
    # Connect to MySQL
    connection = connect_to_mysql()
    if connection is None:
        return None

    cursor = connection.cursor()
    try:
        cursor.execute(f"DESCRIBE {table_name}")
        schema = cursor.fetchall()

        # Return a list of tuples: (column_name, column_type)
        return [(col[0], col[1]) for col in schema]
    except Exception as e:
        print(f"Error fetching schema for table '{table_name}': {e}")
        return None
    finally:
        cursor.close()
        connection.close()

query_rotation_index = 0
def generate_queries_dynamically(table_name):
    queries = generate_random_query_set(table_name)  # Generate a set of random queries
    
    if queries:
        connection = connect_to_mysql()
        if connection is None:
            print("Failed to connect to MySQL to generate queries dyanmaically.")
            return
        cursor = connection.cursor()

        print("\nHere is a set of generated queries:")
        print("-" * 50)
        for i, (query, explanation) in enumerate(queries, 1):
            print(f"Query {i}:\n{query.strip()}")
            print(f"Explanation: {explanation}\n")

            try:
                # Execute the query
                cursor.execute(query.strip())
                results = cursor.fetchall()

                # Display the results (limit to first 5 rows)
                print("Result (Noticing that I am only displaying you preview of the first 5 rows):")
                for row in results[:5]:  # Display only the first 5 rows
                    print(row)

            except Exception as e:
                print(f"Error executing query: {e}")

            print("-" * 50)

        cursor.close()
        connection.close()
    else:
        print("No queries were generated.")

def generate_random_query_set(table_name):
    global query_rotation_index  # Track rotation across calls
    query_set = []

    query_generators = [
    "generate_where_clauses",
    "generate_group_by_clauses",
    "generate_having_clauses", 
    "generate_order_by_clauses",  
    "generate_where_and_having_clauses", 
]

    # Randomize the order of query generators for this execution
    random.shuffle(query_generators)

    # Generate up to 3 random queries per call
    max_queries = 3
    for _ in range(max_queries):
        # Get the current query generator function
        current_generator_name = query_generators[query_rotation_index]
        current_generator = globals().get(current_generator_name)  # Get the function dynamically

        # Rotate to the next generator
        query_rotation_index = (query_rotation_index + 1) % len(query_generators)

        if current_generator:
            # Generate one query and explanation from the current generator
            result = current_generator(table_name, return_random=True)
            if result:
                query, explanation = result
                query_set.append((query, explanation))

    if not query_set:
        print("No queries could be generated.")

    return query_set



### Function for option #3: Generate sample queries using specific language constructs
def handle_specific_construct_option(table_name):
    while True:
        construct_options = {
            "1": "WHERE",
            "2": "GROUP BY",
            "3": "ORDER BY",
            "4": "HAVING",
            "5": "WHERE AND HAVING"
        }
    
        # Display options to the user
        print("\nWhich language constructs would you specifically like to explore?")
        for key, value in construct_options.items():
            print(f"{key}. {value}")
    
        # Prompt user for selection
        construct_choice = input("\nEnter the number corresponding to your choice: ").strip()
    
        # Validate the input
        if construct_choice not in construct_options:
            print("Invalid choice. Please select a valid option from the list.")
            continue
        else:
            break
    
    # Process the choice
    construct_name = construct_options[construct_choice]
    print(f"\nYou selected the {construct_name} construct. Generating queries...")
    
    # Map the choice to the corresponding generator function
    generator_mapping = {
        "WHERE": generate_where_clauses,
        "GROUP BY": generate_group_by_clauses,
        "ORDER BY": generate_order_by_clauses,
        "HAVING": generate_having_clauses,
        "WHERE AND HAVING": generate_where_and_having_clauses
    }
    
    selected_generator = generator_mapping[construct_name]

    # Generate and process the results
    result = selected_generator(table_name, return_random=False, count=3)  # Generate 3 queries
    if result:
        print("\nGenerated Queries:")

        # Connect to the database to execute queries
        connection = connect_to_mysql()
        if not connection:
            print("Failed to connect to the database. Cannot execute queries for handling specific construct.")
            return
        
        cursor = connection.cursor()

        for i, (query, explanation) in enumerate(result, 1):
            print(f"Query {i}:\n{query}")
            print(f"Explanation: {explanation}\n")

            #Executing the query and display results
            try:
                cursor.execute(query)
                query_results = cursor.fetchall()

                print("Results (Noticing that I am only displaying you the preview of the first 5 rows):")
                if query_results:
                    for row in query_results[:5]:  # Display up to 5 rows as a preview
                        print(row)
                else:
                    print("No results found for this query.")
            except Exception as e:
                print(f"Error executing query {i}: {e}")
            print("-" * 50)

        cursor.close()
        connection.close()
    else:
        print(f"Could not generate queries for the {construct_name} construct.")



#Functions for all generators of options #2 and #3
def generate_where_clauses(table_name, return_random=False, count=1):
    engine = create_sqlalchemy_engine()
    connection = connect_to_mysql()
    if connection is None:
        return None

    cursor = connection.cursor()
    try:
        # Fetch schema and column types
        schema = fetch_table_schema(table_name)
        if not schema:
            print("Unable to fetch schema. Cannot generate WHERE clauses.")
            return

        # Infer SQL column types
        query = f"SELECT * FROM {table_name} LIMIT 100;"  # Analyze first 100 rows
        df = pd.read_sql(query, engine)
        column_types = infer_sql_column_types(df)
        unsuitable_columns = identify_unsuitable_columns(df)

        # Filter out unsuitable columns
        categorical_columns = [col for col, sql_type in column_types.items() if sql_type.startswith("VARCHAR") and col not in unsuitable_columns]
        numeric_columns = [col for col, sql_type in column_types.items() if sql_type.startswith(("INT", "DECIMAL")) and col not in unsuitable_columns]

        # Ensure we have at least one suitable column
        if not categorical_columns and not numeric_columns:
            print("No suitable columns found for generating WHERE clauses.")
            return

        # Generate WHERE clauses
        where_clauses = []

        # Example 1: Categorical Equality
        if categorical_columns:
            for col in categorical_columns:
                cursor.execute(f"SELECT DISTINCT {col} FROM {table_name} LIMIT 5")
                distinct_values = [row[0] for row in cursor.fetchall()]
                if distinct_values:
                    random_value = random.choice(distinct_values)
                    escaped_value = str(random_value).replace("'", "''")
                    if random.random() < 0.5:  # 50% chance to use SELECT *
                        where_clauses.append(f"SELECT * FROM {table_name} WHERE {col} = '{escaped_value}';")
                    else:  # 50% chance to select specific columns
                        selected_column = random.choice(categorical_columns)
                        where_clauses.append(f"SELECT {selected_column} FROM {table_name} WHERE {col} = '{escaped_value}';")

        # Example 2: Numeric Comparisons
        if numeric_columns:
            for col in numeric_columns:
                # Fetch min and max values for numeric columns
                cursor.execute(f"SELECT MIN({col}), MAX({col}) FROM {table_name}")
                result = cursor.fetchone()
                if result and all(value is not None for value in result):  # Ensure no NULL values
                    min_val, max_val = result
                    mid_val = (min_val + max_val) // 2  # Calculate a midpoint
                    if random.random() < 0.5:  # 50% chance to use SELECT *
                        where_clauses.append(f"SELECT * FROM {table_name} WHERE {col} > {mid_val};")
                        where_clauses.append(f"SELECT * FROM {table_name} WHERE {col} BETWEEN {min_val} AND {mid_val};")
                    else:  # 50% chance to select specific columns
                        selected_column = random.choice(numeric_columns)
                        where_clauses.append(f"SELECT {selected_column} FROM {table_name} WHERE {col} > {mid_val};")
                        where_clauses.append(f"SELECT {selected_column} FROM {table_name} WHERE {col} BETWEEN {min_val} AND {mid_val};")

        # Example 3: Combined Conditions
        if categorical_columns and numeric_columns:
            random_categorical_col = random.choice(categorical_columns)
            random_numeric_col = random.choice(numeric_columns)
            cursor.execute(f"SELECT DISTINCT {random_categorical_col} FROM {table_name} LIMIT 5")
            distinct_values = [row[0] for row in cursor.fetchall()]
            if distinct_values:
                random_value = random.choice(distinct_values)
                if random.random() < 0.5:  # 50% chance to use SELECT *
                    where_clauses.append(f"""
                    SELECT * FROM {table_name} 
                    WHERE {random_categorical_col} = '{random_value}' 
                    AND {random_numeric_col} > 10;
                    """)
                else:  # 50% chance to select specific columns
                    where_clauses.append(f"""
                    SELECT {random_numeric_col}, {random_categorical_col} FROM {table_name} 
                    WHERE {random_categorical_col} = '{random_value}' 
                    AND {random_numeric_col} > 10;
                    """)

        # Example 4: Logical Operators (e.g., OR)
        if categorical_columns and len(categorical_columns) > 1:
            col1, col2 = random.sample(categorical_columns, 2)
            cursor.execute(f"SELECT DISTINCT {col1} FROM {table_name} LIMIT 5")
            distinct_values1 = [row[0] for row in cursor.fetchall()]
            cursor.execute(f"SELECT DISTINCT {col2} FROM {table_name} LIMIT 5")
            distinct_values2 = [row[0] for row in cursor.fetchall()]
            if distinct_values1 and distinct_values2:
                value1 = random.choice(distinct_values1)
                value2 = random.choice(distinct_values2)
                if random.random() < 0.5:  # 50% chance to use SELECT *
                    where_clauses.append(f"""
                    SELECT * FROM {table_name} 
                    WHERE {col1} = '{value1}' 
                    OR {col2} = '{value2}';
                    """)
                else:  # 50% chance to select specific columns
                    where_clauses.append(f"""
                    SELECT {col1}, {col2} FROM {table_name} 
                    WHERE {col1} = '{value1}' 
                    OR {col2} = '{value2}';
                    """)

        # Limit to 4-5 WHERE clause examples
        #where_clauses = where_clauses[:5]
        if return_random:
            if where_clauses:
                query = random.choice(where_clauses)
                explanation = generate_natural_language_representation_for_where_clauses(query)
                return query.strip(), explanation  # Ensure it returns a tuple
        else:
            selected_clauses = random.sample(where_clauses, min(count, len(where_clauses)))
            return [(clause.strip(), generate_natural_language_representation_for_where_clauses(clause)) for clause in selected_clauses]

    except Exception as e:
        print(f"Error generating WHERE clause examples: {e}")
    finally:
        cursor.close()
        connection.close()

def generate_natural_language_representation_for_where_clauses(query):

    # Split the query into parts for analysis
    tokens = query.strip().split()
    
    # Ensure query has enough parts to analyze
    if len(tokens) < 4:
        return "Unable to generate explanation for this query."

    # Extract SELECT columns and WHERE clause
    select_clause = tokens[1]  # Second word after SELECT
    from_clause = tokens[tokens.index("FROM") + 1]  # Word after FROM
    where_clause_index = tokens.index("WHERE") if "WHERE" in tokens else None

    # Build the explanation
    explanation = ""

    # Handle SELECT clause
    if select_clause == "*":
        explanation += f"Retrieve all columns"
    else:
        explanation += f"Retrieve {select_clause}"

    # Add FROM clause
    explanation += f" from the {from_clause} table"

    # Add WHERE clause if present
    if where_clause_index:
        condition = " ".join(tokens[where_clause_index + 1:])  # Everything after WHERE
        explanation += f" where {condition.replace('=', 'is')}"


        # Return the explanation
        return explanation.capitalize()

def generate_group_by_clauses(table_name, return_random = True, count=1):
    engine = create_sqlalchemy_engine()
    connection = connect_to_mysql()
    if connection is None:
        return None

    cursor = connection.cursor()
    try:
        # Load the table data into a DataFrame
        query = f"SELECT * FROM {table_name} LIMIT 100;"  # Fetch first 100 rows for analysis
        df = pd.read_sql(query, engine)  # Create DataFrame from table

        # Identify unsuitable columns
        unsuitable_columns = identify_unsuitable_columns(df)

        # Fetch schema to identify column types
        schema = fetch_table_schema(table_name)
        if not schema:
            print("Unable to fetch schema. Cannot generate GROUP BY clauses.")
            return

        # Categorize columns
        categorical_columns = [col[0] for col in schema if "char" in col[1].lower() or "text" in col[1].lower() or "varchar" in col[1].lower()]
        numeric_columns = [col[0] for col in schema if "int" in col[1].lower() or "decimal" in col[1].lower() or "float" in col[1].lower()]

        
        # Exclude unsuitable columns
        numeric_columns = [col for col in numeric_columns if col not in unsuitable_columns]
        categorical_columns = [col for col in categorical_columns if col not in unsuitable_columns]

        if not categorical_columns or not numeric_columns:
            print("Insufficient columns for generating GROUP BY examples.")
            return

        # Generate GROUP BY clauses
        group_by_clauses = []

        # Example 1: SUM Aggregation
        if numeric_columns and categorical_columns:
            random_categorical_col = random.choice(categorical_columns)  # Randomly select a categorical column
            random_numeric_col = random.choice(numeric_columns)  # Randomly select a numeric column
            group_by_clauses.append(f"""
                SELECT {random_categorical_col}, SUM({random_numeric_col}) AS total_value
                FROM {table_name}
                GROUP BY {random_categorical_col};
            """)

        # Example 2: COUNT Aggregation
        if categorical_columns:
            random_categorical_col = random.choice(categorical_columns)
            group_by_clauses.append(f"""
                SELECT {random_categorical_col}, COUNT(*) AS count_value
                FROM {table_name}
                GROUP BY {random_categorical_col};
            """)

        # Example 3: SUM with a Derived Calculation
        if len(numeric_columns) > 1 and categorical_columns:
            random_categorical_col = random.choice(categorical_columns)
            numeric_col_1 = random.choice(numeric_columns)
            numeric_col_2 = random.choice([col for col in numeric_columns if col != numeric_col_1])
            group_by_clauses.append(f"""
                SELECT {random_categorical_col}, SUM({numeric_col_1} * {numeric_col_2}) AS calculated_value
                FROM {table_name}
                GROUP BY {random_categorical_col};
            """)

        # Example 4: AVERAGE Aggregation
        if numeric_columns and categorical_columns:
            random_categorical_col = random.choice(categorical_columns)
            random_numeric_col = random.choice(numeric_columns)
            group_by_clauses.append(f"""
                SELECT {random_categorical_col}, AVG({random_numeric_col}) AS avg_value
                FROM {table_name}
                GROUP BY {random_categorical_col};
            """)

        # If no group_by_clauses were generated, return None
        if not group_by_clauses:
            print("No GROUP BY queries could be generated.")
            return None, None
        
        # If return_random=True, select one random clause and return it
        if return_random:
            if group_by_clauses:
                query = random.choice(group_by_clauses).strip()
                explanation = generate_natural_language_representation_for_group_by_clauses(query)
                return query, explanation  # Ensure it returns a tuple
            else:
                return None  # No query was generated
            
        # Generate multiple queries if requested
        selected_clauses = random.sample(group_by_clauses, min(count, len(group_by_clauses)))
        return [(clause.strip(), generate_natural_language_representation_for_group_by_clauses(clause)) for clause in selected_clauses]
        

    except Exception as e:
        print(f"Error generating GROUP BY clause examples: {e}")
    finally:
        cursor.close()
        connection.close()

def generate_natural_language_representation_for_group_by_clauses(query):
    query = query.strip().lower()  # Normalize query for processing
    explanation = ""

    # Extract components
    if "select" in query and "group by" in query:
        # Identify SELECT columns
        select_clause = query.split("from")[0].replace("select", "").strip()
        columns = [col.strip() for col in select_clause.split(",")]

        # Extract the GROUP BY column
        group_by_clause = query.split("group by")[1].strip().split(";")[0]
        group_by_column = group_by_clause.strip()

        # Determine the aggregate function used
        if "sum(" in select_clause:
            agg_column = select_clause.split("sum(")[1].split(")")[0]
            explanation = f"Calculate the total value of {agg_column} grouped by {group_by_column}."
        elif "avg(" in select_clause:
            agg_column = select_clause.split("avg(")[1].split(")")[0]
            explanation = f"Calculate the average value of {agg_column} grouped by {group_by_column}."
        elif "count(" in select_clause:
            explanation = f"Count the number of records grouped by {group_by_column}."
        elif "*" in select_clause:
            explanation = f"Retrieve all records grouped by {group_by_column}."
        else:
            explanation = f"Perform grouping based on {group_by_column}."

    return explanation if explanation else "Unable to generate explanation for this query."

def generate_order_by_clauses(table_name, return_random=False, count=1):
    engine = create_sqlalchemy_engine()
    connection = connect_to_mysql()
    if connection is None:
        return None

    cursor = connection.cursor()
    try:
        # Fetch schema and column types
        query = f"SELECT * FROM {table_name} LIMIT 100;"  # Analyze first 100 rows
        df = pd.read_sql(query, engine)
        column_types = infer_sql_column_types(df)
        unsuitable_columns = identify_unsuitable_columns(df)

        # Filter out unsuitable columns
        sortable_columns = [col for col in df.columns if col not in unsuitable_columns]

        # Ensure there are sortable columns
        if not sortable_columns:
            print("No suitable columns found for generating ORDER BY clauses.")
            return None

        # Generate ORDER BY clauses
        order_by_clauses = []

        for _ in range(count):  # Generate the required number of queries
            random_column = random.choice(sortable_columns)  # Randomly pick a sortable column
            # Randomly select columns to include in SELECT
            selected_columns = random.sample(sortable_columns, random.randint(1, len(sortable_columns)))
            select_clause = ", ".join(selected_columns) if random.random() < 0.7 else "*"
            # Randomly decide the order direction
            order = random.choice(["ASC", "DESC"])
            order_by_clauses.append(f"SELECT {select_clause} FROM {table_name} ORDER BY {random_column} {order};")

        # If return_random=True, select one random clause and return it
        if return_random:
            if order_by_clauses:
                query = random.choice(order_by_clauses).strip()
                explanation = generate_natural_language_representation_for_order_by_clauses(query)
                return query, explanation  # Ensure it returns a tuple
            else:
                return None  # No query was generated
            
        # For multiple queries, return them with explanations
        else:
            return [
            (clause.strip(), generate_natural_language_representation_for_order_by_clauses(clause))
            for clause in order_by_clauses
        ]

    except Exception as e:
        print(f"Error generating ORDER BY clause examples: {e}")
    finally:
        cursor.close()
        connection.close()

def generate_natural_language_representation_for_order_by_clauses(query):
    query = query.strip().lower()  # Normalize query for processing

    # Remove any trailing semicolon from the query
    if query.endswith(";"):
        query = query[:-1]

    # Extract components
    if "order by" in query:

        # Extract SELECT clause
        select_clause = query.split("from")[0].replace("select", "").strip()
        selected_columns = [col.strip() for col in select_clause.split(",")]

        # Identify the column and order direction
        order_by_clause = query.split("order by")[1].strip()
        column, direction = order_by_clause.split()[:2]  # Get column and direction (e.g., ASC/DESC)
        
        # Map ASC/DESC to natural language terms
        if direction == "asc":
            direction_text = "ascending"
        elif direction == "desc":
            direction_text = "descending"
        else:
            direction_text = direction  # Fallback to the original direction if unknown
        

        # Build the explanation dynamically
        if "*" in selected_columns:
            explanation = (
                f"Display all columns in the table by sorting '*' in {direction_text} order."
            )
        elif len(selected_columns) == 1:
            explanation = (
                f"Display {selected_columns[0]} in the table by sorting {selected_columns} in {direction_text} order."
            )
        elif len(selected_columns) == 2:
            explanation = (
                f"Display {selected_columns[0]} and {selected_columns[1]} in the table by sorting {selected_columns} in {direction_text} order."
            )
        else:
            formatted_columns = ", ".join(selected_columns[:-1]) + f", and {selected_columns[-1]}"
            explanation = (
                f"Display {formatted_columns} from the table, sorted by {selected_columns} in {direction_text} order."
            )
    else:
        explanation = "Unable to generate explanation for this query."

    return explanation

def generate_having_clauses(table_name, return_random=True, count=1):

    engine = create_sqlalchemy_engine()
    connection = connect_to_mysql()
    if connection is None:
        return None

    cursor = connection.cursor()
    try:
        # Load the table data into a DataFrame
        query = f"SELECT * FROM {table_name} LIMIT 100;"  # Fetch first 100 rows for analysis
        df = pd.read_sql(query, engine)  # Create DataFrame from the table

        # Identify unsuitable columns
        unsuitable_columns = identify_unsuitable_columns(df)

        # Fetch schema to identify column types
        schema = fetch_table_schema(table_name)
        if not schema:
            print("Unable to fetch schema. Cannot generate HAVING clauses.")
            return

        # Categorize columns
        numeric_columns = [col[0] for col in schema if "int" in col[1].lower() or "decimal" in col[1].lower() or "float" in col[1].lower()]
        numeric_columns = [col for col in numeric_columns if col not in unsuitable_columns]
        if not numeric_columns:
            print("No numeric columns available for generating HAVING clauses.")
            return None

        # Generate HAVING clauses
        having_clauses = []

        # Example 1: HAVING SUM
        if numeric_columns:
            random_numeric_col = random.choice(numeric_columns)
            having_clauses.append(f"""
                SELECT {random_numeric_col}, SUM({random_numeric_col}) AS total_value
                FROM {table_name}
                GROUP BY {random_numeric_col}
                HAVING SUM({random_numeric_col}) > 100;
            """)

        # Example 2: HAVING AVG
        if numeric_columns:
            random_numeric_col = random.choice(numeric_columns)
            having_clauses.append(f"""
                SELECT {random_numeric_col}, AVG({random_numeric_col}) AS avg_value
                FROM {table_name}
                GROUP BY {random_numeric_col}
                HAVING AVG({random_numeric_col}) < 50;
            """)

        # Example 3: HAVING COUNT
        if numeric_columns:
            random_numeric_col = random.choice(numeric_columns)
            having_clauses.append(f"""
                SELECT {random_numeric_col}, COUNT(*) AS count_value
                FROM {table_name}
                GROUP BY {random_numeric_col}
                HAVING COUNT(*) > 10;
            """)

        # If no HAVING clauses were generated, return None
        if not having_clauses:
            print("No HAVING queries could be generated.")
            return None, None

        # If return_random=True, select one random clause and return it
        if return_random:
            query = random.choice(having_clauses).strip()
            explanation = generate_natural_language_representation_for_having_clauses(query)
            return query, explanation  # Return the query and its explanation
        else:
            # Handle return_random=False for multiple queries
            selected_clauses = random.sample(having_clauses, min(count, len(having_clauses)))
            return [
                (clause.strip(), generate_natural_language_representation_for_having_clauses(clause))
                for clause in selected_clauses
            ]

    except Exception as e:
        print(f"Error generating HAVING clause examples: {e}")
    finally:
        cursor.close()
        connection.close()

def generate_natural_language_representation_for_having_clauses(query):
    """
    Generate a natural language explanation for a HAVING SQL query.
    """
    query = query.strip().lower()  # Normalize query for processing
    explanation = ""

    # Extract components
    if "select" in query and "having" in query:
        # Extract SELECT and HAVING parts
        select_clause = query.split("from")[0].replace("select", "").strip()
        having_clause = query.split("having")[1].strip()

        # Identify aggregate function and column
        if "sum(" in select_clause:
            agg_column = select_clause.split("sum(")[1].split(")")[0]
            explanation = f"Retrieve groups where the total value of {agg_column} satisfies the condition: {having_clause}."
        elif "avg(" in select_clause:
            agg_column = select_clause.split("avg(")[1].split(")")[0]
            explanation = f"Retrieve groups where the average value of {agg_column} satisfies the condition: {having_clause}."
        elif "count(" in select_clause:
            explanation = f"Retrieve groups where the count of rows satisfies the condition: {having_clause}."

    return explanation if explanation else "Unable to generate explanation for this query."

def generate_where_and_having_clauses(table_name, return_random=True, count=1):

    engine = create_sqlalchemy_engine()
    connection = connect_to_mysql()
    if connection is None:
        return None

    cursor = connection.cursor()
    try:
        # Load the table data into a DataFrame
        query = f"SELECT * FROM {table_name} LIMIT 100;"  # Fetch first 100 rows for analysis
        df = pd.read_sql(query, engine)  # Create DataFrame from the table

        # Identify unsuitable columns
        unsuitable_columns = identify_unsuitable_columns(df)

        # Fetch schema to identify column types
        schema = fetch_table_schema(table_name)
        if not schema:
            print("Unable to fetch schema. Cannot generate WHERE and HAVING clauses.")
            return None

        # Categorize columns
        numeric_columns = [col[0] for col in schema if "int" in col[1].lower() or "decimal" in col[1].lower() or "float" in col[1].lower()]
        categorical_columns = [col[0] for col in schema if "char" in col[1].lower() or "varchar" in col[1].lower() or "text" in col[1].lower()]
        numeric_columns = [col for col in numeric_columns if col not in unsuitable_columns]
        categorical_columns = [col for col in categorical_columns if col not in unsuitable_columns]

        if not numeric_columns or not categorical_columns:
            print("Insufficient columns for generating WHERE and HAVING clauses.")
            return None

        # Generate WHERE and HAVING clauses
        where_and_having_clauses = []

        # Example 1: WHERE with categorical condition, HAVING with SUM
        if numeric_columns and categorical_columns:
            random_categorical_col = random.choice(categorical_columns)
            cursor.execute(f"SELECT DISTINCT {random_categorical_col} FROM {table_name} LIMIT 10")
            distinct_values = [row[0] for row in cursor.fetchall() if row[0] is not None]
            if distinct_values:
                random_value = random.choice(distinct_values)
                random_numeric_col = random.choice(numeric_columns)
                where_and_having_clauses.append(f"""
                    SELECT {random_numeric_col}, SUM({random_numeric_col}) AS total_value
                    FROM {table_name}
                    WHERE {random_categorical_col} = '{random_value}'
                    GROUP BY {random_numeric_col}
                    HAVING SUM({random_numeric_col}) > 100;
                """)

        # Example 2: WHERE with numeric range, HAVING with AVG
        if numeric_columns:
            random_numeric_col = random.choice(numeric_columns)
            cursor.execute(f"SELECT MIN({random_numeric_col}), MAX({random_numeric_col}) FROM {table_name}")
            min_val, max_val = cursor.fetchone()
            if min_val is not None and max_val is not None and min_val != max_val:
                mid_val = (min_val + max_val) // 2
                where_and_having_clauses.append(f"""
                    SELECT {random_numeric_col}, AVG({random_numeric_col}) AS avg_value
                    FROM {table_name}
                    WHERE {random_numeric_col} BETWEEN {min_val} AND {mid_val}
                    GROUP BY {random_numeric_col}
                    HAVING AVG({random_numeric_col}) < {mid_val};
                """)

        # Example 3: WHERE with multiple conditions, HAVING with COUNT
        if numeric_columns and categorical_columns:
            random_categorical_col = random.choice(categorical_columns)
            cursor.execute(f"SELECT DISTINCT {random_categorical_col} FROM {table_name} LIMIT 10")
            distinct_values = [row[0] for row in cursor.fetchall() if row[0] is not None]
            if distinct_values:
                random_value = random.choice(distinct_values)
                random_numeric_col = random.choice(numeric_columns)
                where_and_having_clauses.append(f"""
                    SELECT {random_numeric_col}, COUNT(*) AS count_value
                    FROM {table_name}
                    WHERE {random_categorical_col} = '{random_value}' AND {random_numeric_col} > 20
                    GROUP BY {random_numeric_col}
                    HAVING COUNT(*) > 5;
                """)

        # If no queries were generated, return None
        if not where_and_having_clauses:
            print("No WHERE and HAVING queries could be generated.")
            return None, None

        # If return_random=True, select one random clause and return it
        if return_random:
            query = random.choice(where_and_having_clauses).strip()
            explanation = generate_natural_language_representation_for_where_and_having_clauses(query)
            return query, explanation  # Return the query and its explanation

        else:
            # Handle return_random=False for multiple queries
            selected_clauses = random.sample(where_and_having_clauses, min(count, len(where_and_having_clauses)))
            return [
                (clause.strip(), generate_natural_language_representation_for_where_and_having_clauses(clause))
                for clause in selected_clauses
            ]

    except Exception as e:
        print(f"Error generating WHERE and HAVING clause examples: {e}")
    finally:
        cursor.close()
        connection.close()

def generate_natural_language_representation_for_where_and_having_clauses(query):
    """
    Generate a natural language explanation for a combined WHERE and HAVING SQL query.
    """
    query = query.strip().lower()  # Normalize query for processing
    explanation = ""

    # Extract components
    if "select" in query and "having" in query and "where" in query:
        # Extract WHERE and HAVING parts
        select_clause = query.split("from")[0].replace("select", "").strip()
        where_clause = query.split("where")[1].split("group by")[0].strip()
        having_clause = query.split("having")[1].strip()

        # Explanation for SELECT and WHERE
        explanation = f"Retrieve {select_clause} where {where_clause}."

        # Add HAVING explanation
        if "sum(" in select_clause:
            agg_column = select_clause.split("sum(")[1].split(")")[0]
            explanation += f" Group the results by {agg_column} and filter groups where {having_clause}."
        elif "avg(" in select_clause:
            agg_column = select_clause.split("avg(")[1].split(")")[0]
            explanation += f" Group the results by {agg_column} and filter groups where {having_clause}."
        elif "count(" in select_clause:
            explanation += f" Group the results and filter groups where {having_clause}."

    return explanation if explanation else "Unable to generate explanation for this query."



### Handle specific query when users ask specific questions in natural language
def handle_specific_query(table_name):

    # Step 1: Explain supported queries
    print("\nChatDB currently only supports the following 5 SQL query types:")
    while True:
        query_types = {
        "1": ("WHERE", "Filters rows based on specified conditions."),
        "2": ("GROUP BY", "Groups rows that have the same values in specified columns."),
        "3": ("ORDER BY", "Sorts rows by specified columns in ascending or descending order."),
        "4": ("HAVING", "Filters groups of rows based on aggregate conditions (sum or average)."),
        "5": ("WHERE AND HAVING", "Combines row filtering and group filtering in a single query.")
        }

        for key, (name, description) in query_types.items():
            print(f"{key}. {name}: {description}")

        # Step 2: Prompt user to choose a query type
        query_choice = input("\nEnter the number corresponding to the query type you'd like to ask: ").strip()

        if query_choice not in query_types:
            print("Invalid choice. Please select a valid option.")
            continue
        else:
            break

    # Step 3: Guide based on the selected query type
    query_type, description = query_types[query_choice]
    print(f"\nYou selected the {query_type} query type.")

    if query_type == "WHERE":
        # Call WHERE-specific handling
        handle_where_query(table_name)
    elif query_type == "GROUP BY":
        handle_group_by_query(table_name)
    elif query_type == "ORDER BY":
        handle_order_by_query(table_name)
    elif query_type == "HAVING":
        handle_having_query(table_name)
    elif query_type == "WHERE AND HAVING":
        handle_where_and_having_query(table_name)

def handle_where_query(table_name):
    print("\nPlease use the following template:")
    print("find <A> where <B> is <C>")
    print("\nReplace the letter in the template of your choice.")
    print("\nA: the column name you want to display")
    print("\nB: the column name of your chosen condition")
    print("\nC: a specific value under column B")

    user_query = input("\nEnter your WHERE query: ").strip().lower()

    if "find" in user_query and "where" in user_query and "is" in user_query:
        try:
            # Extract query components
            find_part = user_query.split("where")[0].strip()
            column1 = find_part.split("find")[1].strip()

            where_part = user_query.split("where")[1].strip()
            column2, specific_value = where_part.split("is")[0].strip(), where_part.split("is")[1].strip()

            # Validate and process
            if validate_columns_and_value(table_name, column1, column2, value=specific_value):
                sql_query = f"SELECT {column1} FROM {table_name} WHERE {column2} = '{specific_value}';"
                print(f"\nGenerated Query:\n{sql_query}")
                execute_and_display_query(sql_query)
            else:
                print("Validation failed. Please check your entered column names exactly matching with the column names on your file.")
        except Exception as e:
            print(f"Error processing your WHERE query. Ensure it follows the template. Details: {e}")
    else:
        print("Invalid query structure. Please follow the template: find find <A> where <B> is <C>.")

def handle_group_by_query(table_name):
    print("\nPlease use the following template:")
    print("Template: find <A> (<B>) group by <C>")
    print("\nReplace the letter in the template of your choice.")
    print("\nA: the column name you want to display")
    print("\nB: the column name you would like to sum or average")
    print("\nC: column name that you would like to group by")

    user_query = input("\nEnter your GROUP BY query: ").strip().lower()

    if "find" in user_query and "group by" in user_query:
        try:
            # Extract query components
            find_part = user_query.split("group by")[0].strip()
            aggregate_function, column1 = find_part.split("(")[0].split("find")[1].strip(), find_part.split("(")[1].strip(")")

            group_by_column = user_query.split("group by")[1].strip()

            # Validate and process
            if validate_columns_and_value(table_name, column1, group_by_column):
                sql_query = f"SELECT {group_by_column}, {aggregate_function.upper()}({column1}) FROM {table_name} GROUP BY {group_by_column};"
                print(f"\nGenerated Query:\n{sql_query}")
                execute_and_display_query(sql_query)
            else:
                print("Validation failed. Please make sure the column names you enter exactly match the column names in your file.")
        except Exception as e:
            print(f"Error processing your GROUP BY query. Ensure it follows the template. Details: {e}")
    else:
        print("Invalid query structure. Please follow the template: find <aggregate function>(<column name1>) group by <column name2>.")

def handle_order_by_query(table_name):
    print("\nPlease use the following template:")
    print("Template: order <A> by <B>")
    print("\nReplace the letter in the template of your choice.")
    print("\nA: the column name you want to order")
    print("\nB: 'asc' as ascending order or 'desc' as descending order")

    user_query = input("\nEnter your ORDER BY query: ").strip().lower()

    if "order" in user_query and "by" in user_query:
        try:
            # Extract query components
            column, direction = user_query.split("order")[1].split("by")
            column = column.strip()
            direction = direction.strip()

            if direction not in ["asc", "desc"]:
                print("Invalid direction. Use 'asc' or 'desc'.")
                return

            # Validate and process
            if validate_columns_and_value(table_name, column):
                sql_query = f"SELECT * FROM {table_name} ORDER BY {column} {direction.upper()};"
                print(f"\nGenerated Query:\n{sql_query}")
                execute_and_display_query(sql_query)
            else:
                print("Validation failed. Please check your column name.")
        except Exception as e:
            print(f"Error processing your ORDER BY query. Ensure it follows the template. Details: {e}")
    else:
        print("Invalid query structure. Please follow the template: order <column name> by <asc/desc>.")

def handle_having_query(table_name):
    print("\nPlease use the following template:")
    print("Teplate: find <A>(<B>) group by <C> having <D>(<E>)<F><G>")
    print("\nReplace the letter in the template of your choice.")
    print("\nA: 'sum' or 'avg' as average")
    print("\nB: sum or average of the column name you want")
    print("\nC: column name that you would like to group by")
    print("\nD: 'sum' or 'avg' as average")
    print("\nE: the column name you would like to sum or average")
    print("\nF: '=' '<' or '>'")
    print("\nG: a specific condition value you want to generate(Could be a number)")

    user_query = input("\nEnter your HAVING query: ").strip().lower()

    if "find" in user_query and "group by" in user_query and "having" in user_query:
        try:
            # Extract query components
            find_part = user_query.split("group by")[0].strip()
            aggregate_function, column1 = find_part.split("(")[0].split("find")[1].strip(), find_part.split("(")[1].strip(")")

            group_by_part = user_query.split("group by")[1].split("having")[0].strip()
            having_part = user_query.split("having")[1].strip()
            having_aggregate_function, having_column, operator, having_value = (
                having_part.split("(")[0].strip(),
                having_part.split("(")[1].split(")")[0],
                having_part.split(")")[1].split()[0],
                having_part.split(")")[1].split()[1],
            )

            # Validate and process
            if validate_columns_and_value(
                table_name, column1, group_by_part, having_column, having_value=having_value
            ):
                sql_query = f"""
                SELECT {group_by_part}, {aggregate_function.upper()}({column1}) 
                FROM {table_name} 
                GROUP BY {group_by_part} 
                HAVING {having_aggregate_function.upper()}({having_column}) {operator} {having_value};
                """
                print(f"\nGenerated Query:\n{sql_query}")
                execute_and_display_query(sql_query)
            else:
                print("Validation failed. Please check your column names or value.")
        except Exception as e:
            print(f"Error processing your HAVING query. Ensure it follows the template. Details: {e}")
    else:
        print("Invalid query structure. Please follow the template: 'find <aggregate function>(<column name>) group by <column name> having <aggregate function>(<column name>) <operator> <value>'.")

def handle_where_and_having_query(table_name):

    print("\nPlease use the following template:")
    print("Teplate: find <A>(<B>) where <C> is <D> group by <E> having <F> (<G>) <H>")
    print("\nReplace the letter in the template of your choice.")
    print("\nA: 'sum' or 'avg' as average")
    print("\nB: sum or average of the column name you want")
    print("\nC: the column name of your chosen condition")
    print("\nD: a specific value of your chosen column")
    print("\nE: column name that you would like to group by")
    print("\nF: 'sum' or 'avg' as average")
    print("\nG: the column name you would like to sum or average")
    print("\nH: a specific condition value you want to generate(Could be a number")

    user_query = input("\nEnter your WHERE and HAVING query: ").strip().lower()

    if "find" in user_query and "where" in user_query and "group by" in user_query and "having" in user_query:
        try:
            # Parse components
            find_part = user_query.split("where")[0].strip()
            aggregate_function = find_part.split("find")[1].split("(")[0].strip()
            column1 = find_part.split("(")[1].split(")")[0].strip()

            where_part = user_query.split("where")[1].split("group by")[0].strip()
            where_column = where_part.split("is")[0].strip()
            where_value = where_part.split("is")[1].strip()

            group_by_part = user_query.split("group by")[1].split("having")[0].strip()

            having_part = user_query.split("having")[1].strip()
            having_aggregate_function = having_part.split("(")[0].strip()
            having_column = having_part.split("(")[1].split(")")[0].strip()
            operator = having_part.split(")")[1].strip().split()[0]
            value = having_part.split(")")[1].strip().split()[1]

            # Validate and process
            if validate_columns_and_value(table_name, column1, where_column, group_by_part, having_column, value=value, skip_value_check=True):
                sql_query = f"""
                SELECT {group_by_part}, {aggregate_function.upper()}({column1}) 
                FROM {table_name} 
                WHERE {where_column} = '{where_value}' 
                GROUP BY {group_by_part} 
                HAVING {having_aggregate_function.upper()}({having_column}) {operator} {value};
                """
                print(f"\nGenerated Query:\n{sql_query}")
                execute_and_display_query(sql_query)
            else:
                print("Validation failed. Please check your column names or value.")
        except Exception as e:
            print(f"Error processing your WHERE and HAVING query. Ensure it follows the template. Details: {e}")
    else:
        print("Invalid query structure. Please follow the template: 'find <aggregate function>(<column name>) where <column name> is <value> group by <column name> having <aggregate function>(<column name>) <operator> <value>'.")

def validate_columns_and_value(table_name, *columns, value=None, skip_value_check=False, having_value=None):
    """
    Validate if the columns exist in the table and if the value exists in a specific column.
    """
    connection = connect_to_mysql()
    if connection is None:
        print("Failed to connect to the database.")
        return False

    cursor = connection.cursor()
    try:
        # Validate columns
        cursor.execute(f"SHOW COLUMNS FROM {table_name};")
        table_columns = [row[0] for row in cursor.fetchall()]
        for column in columns:
            normalized_column = column.replace(" ", "_")
            if normalized_column not in table_columns:
                print(f"Column '{column}' (normalized as '{normalized_column}') does not exist in the table.")
                return False
        
        # Validate WHERE value
        if not skip_value_check and value is not None and len(columns) > 0:
            # Check if the value exists in the specific column
            first_column = columns[-1]
            query = f"SELECT COUNT(*) FROM {table_name} WHERE {first_column} = %s;"
            cursor.execute(query, (value,))
            if cursor.fetchone()[0] == 0:
                print(f"Value '{value}' does not exist in column '{first_column}'.")
                return False

        # Validate HAVING value
        if having_value is not None:
            # HAVING validation should focus only on column existence, not specific value checks
            having_column = columns[-1]  # Assume the last column is used in HAVING
            if having_column not in table_columns:
                print(f"Column '{having_column}' does not exist for HAVING condition.")
                return False

        return True
    except Exception as e:
        print(f"Error validating columns and value: {e}")
        return False
    finally:
        cursor.close()
        connection.close()

def execute_and_display_query(sql_query):
    """
    Execute the given SQL query and display the results.
    """
    connection = connect_to_mysql()
    if connection is None:
        print("Failed to connect to the database. Cannot execute query.")
        return

    cursor = connection.cursor()
    try:
        cursor.execute(sql_query)
        results = cursor.fetchall()
        print("\nResults (showing up to 5 rows):")
        for row in results[:5]:  # Display up to 5 rows
            print(row)
    except Exception as e:
        print(f"Error executing query: {e}")
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":

    print("Welcome to ChatDB!")

    # User uploading their file/files
    file_path = input("Please upload your CSV file you would like to explore on: ").strip()
    table_name = check_file_then_upload(file_path)

    if table_name:
        handle_user_input(file_path, table_name)
    else:
        print("Failed for processing the action:(")