CREATE TABLE banks (
    bank_id NUMBER PRIMARY KEY,
    bank_name VARCHAR2(255) UNIQUE NOT NULL,
    location VARCHAR2(255),
    established_year NUMBER
);