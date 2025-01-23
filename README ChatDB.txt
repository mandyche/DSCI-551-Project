README: ChatDB 


1. Overview
This folder contains the necessary files for running the ChatDB application, which allows users to upload .csv files, process them into a MySQL database, and explore SQL queries interactively through natural language inputs or specific constructs. Follow the instructions below to set up and run the application.



2. File Structure
ChatDB.py: The main Python script containing the full implementation of the ChatDB application.
Coffee_Shop_Sales.csv: A sample dataset for testing the application. Users can upload this file or any valid .csv file of their choice.

README ChatDB.txt: This file, providing instructions to run the program and details about the folder structure.



3. Requirements
To run the program, ensure the following dependencies and environment setups:

	Python 3.7+
	Required Python libraries:
		pandas
		mysql-connector-python
		pymysql
		SQLAlchemy
	MySQL Database:
		A local MySQL server running on localhost with:
			Username: root
			Password: MC1007unicornlay (adjust if using a different password).
			Database: chatdb (automatically created if it doesn’t exist).
			VS Code or any terminal with Python installed.



4. How to Run the Program
Set Up MySQL:

	Ensure your MySQL server is running on localhost.
	Update the credentials in the ChatDB.py script if they differ from your setup.

Run the Script:
	Open a terminal or command prompt.
	Navigate to the directory where ChatDB.py is located.
	Execute the script using the following command:
		python ChatDB.py
	Upload a Dataset:

		The program will prompt you to upload a .csv file.
		Use the provided Coffee_Shop_Sales.csv file or any other valid .csv dataset.
Explore the Database:
	Follow the on-screen prompts to explore your data using:
	Sample queries.
	Specific SQL constructs (e.g., WHERE, GROUP BY, ORDER BY).
	Natural language queries.



5. File Descriptions
ChatDB.py:
	Implements the ChatDB application, including:
		Uploading and processing .csv files.
		Creating MySQL tables and inserting data.
		Generating SQL queries and handling natural language input.
		Outputs query results directly in the terminal.

Coffee_Shop_Sales.csv:
	A test dataset for verifying the program’s functionality.
	Contains sales data for a coffee shop, useful for exploring SQL constructs like filtering, grouping, and ordering.

README ChatDB.txt:
	Provides instructions on setting up, running the program, and understanding the file structure.



6. Notes
Ensure that the pandas, pymysql, mysql-connector-python, and SQLAlchemy libraries are installed before running the program.

The MySQL database credentials (host, user, password) are hardcoded in ChatDB.py. Update them if needed for your local setup.

Use the provided test dataset (.csv file) for exploration.