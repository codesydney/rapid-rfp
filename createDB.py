import sqlite3

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('rfp.db')

# Create a cursor object to execute SQL commands
cursor = conn.cursor()

# SQL command to create the rfp table
create_table_query = '''
CREATE TABLE IF NOT EXISTS rfp (
    ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Query TEXT NOT NULL,
    Confidence REAL,
    Answer TEXT
);
'''

# Execute the SQL command to create the table
cursor.execute(create_table_query)

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Table 'rfp' created successfully in 'rfp.db'.")