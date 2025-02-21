import sqlite3
import csv
import random

# Function to read the CSV file and return the data as a list of dictionaries
def read_csv(file_path):
    data = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='|')
        for row in reader:
            data.append(row)
    return data

# Function to update rows with dummy answers and confidence scores
def update_with_dummy_data(data):
    for row in data:
        row['Answer'] = f"Dummy Answer for Query ID {row['ID']}"  # Add dummy answer
        row['Confidence'] = random.randint(1, 9)  # Add random confidence score (1-9)
    return data

# Function to write updated data back to the CSV file
def write_csv(data, file_path):
    fieldnames = ['ID', 'Query', 'Confidence', 'Answer']
    with open(file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='|')
        writer.writeheader()
        writer.writerows(data)

# Function to insert or update data in the SQLite database
def insert_or_update_db(data, db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    for row in data:
        # Check if the ID already exists in the database
        cursor.execute('SELECT * FROM rfp WHERE ID = ?', (row['ID'],))
        existing_record = cursor.fetchone()

        if existing_record:
            # Update the existing record
            cursor.execute('''
                UPDATE rfp
                SET Query = ?, Confidence = ?, Answer = ?
                WHERE ID = ?
            ''', (row['Query'], row['Confidence'], row['Answer'], row['ID']))
            print(f"Updated record with ID {row['ID']} in the database.")
        else:
            # Insert a new record
            cursor.execute('''
                INSERT INTO rfp (ID, Query, Confidence, Answer)
                VALUES (?, ?, ?, ?)
            ''', (row['ID'], row['Query'], row['Confidence'], row['Answer']))
            print(f"Inserted new record with ID {row['ID']} into the database.")

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

# Main function
def main():
    # Path to the CSV file and SQLite database
    csv_file_path = 'rfp.csv'
    db_file_path = 'rfp.db'

    # Step 1: Read the CSV file
    data = read_csv(csv_file_path)
    print("CSV data read successfully.")

    # Step 2: Update with dummy answers and confidence scores
    updated_data = update_with_dummy_data(data)
    print("Updated data with dummy answers and confidence scores.")

    # Step 3: Write updated data back to the CSV file
    write_csv(updated_data, csv_file_path)
    print("Updated data written back to the CSV file.")

    # Step 4: Insert or update data in the SQLite database
    insert_or_update_db(updated_data, db_file_path)
    print("Data insertion/update in the SQLite database completed.")

# Run the program
if __name__ == '__main__':
    main()