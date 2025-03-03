import mysql.connector
import datetime

def connectDatabase():
    connection = mysql.connector.connect(
        host="localhost",
        port="3306",
        user="root",
        password="123456",
        database="face_attendance"
    )
    return connection


def createTables():
    con = connectDatabase()
    cursor = con.cursor()
    create_people_table = """
    CREATE TABLE IF NOT EXISTS people (
        id INT PRIMARY KEY,
        name VARCHAR(100) NOT NULL
    );
     """
    cursor.execute(create_people_table)

    create_attendance_table = """ 
    CREATE TABLE IF NOT EXISTS attendance(
        id INT AUTO_INCREMENT PRIMARY KEY,
        idPeople INT,
        timeCheckin DATETIME,
        timeCheckout DATETIME,
        FOREIGN KEY (idPeople) REFERENCES people(id)
    );
    """
    cursor.execute(create_attendance_table)
    cursor.close()
    con.close()

createTables()


def insertOrUpdate(id,name):
    con = connectDatabase()
    query = "SELECT * FROM people WHERE id = %s"
    cursor = con.cursor()
    cursor.execute(query, (id,))
    records = cursor.fetchall()
    isRecord = 0
    for row in records:
        isRecord = 1
    if isRecord == 0:
        query = "INSERT INTO people (id, name) VALUES (%s,%s)"
        cursor.execute(query,(id,name))
    else:
        query = "UPDATE people SET name = %s WHERE id = %s"
        cursor.execute(query,(name,id))
    con.commit()
    con.close()
    cursor.close()



def checkInAndCheckOut(idPeople):
    check = False
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    con = connectDatabase()
    cursor = con.cursor()
    query = "SELECT * FROM attendance WHERE idPeople = %s AND DATE(timeCheckin) = %s"
    cursor.execute(query,(idPeople,current_date))
    records = cursor.fetchall()

    #Kiểm tra bản ghi checkin
    isRecorded = len(records) > 0

    if not isRecorded:
        query = "INSERT INTO attendance (idPeople, timeCheckin, timeCheckout) VALUES (%s, %s, %s)"
        cursor.execute(query, (idPeople, current_time, None))
        check = True
        print("Check-in recorded successfully")
    else:
        query = "UPDATE attendance SET timeCheckout = %s WHERE idPeople = %s AND DATE(timeCheckin) = %s"
        cursor.execute(query, (current_time,idPeople,current_date))
        check = False
        print('Check-out recorded successfully')

    con.commit()
    cursor.close()
    con.close()
    return check

def getProfile(id):
    con = connectDatabase()
    query = "SELECT * FROM people WHERE id = %s"
    cursor = con.cursor()
    cursor.execute(query,(id,))
    records = cursor.fetchall()
    profile = None
    for row in records:
        profile = row
    con.close()
    return profile