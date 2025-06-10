CREATE TABLE reviews (
    review_id NUMBER PRIMARY KEY,
    bank_id NUMBER,
    bank_name VARCHAR2(255) NOT NULL,
    rating NUMBER CHECK (rating BETWEEN 1 AND 5),
    review_date DATE DEFAULT SYSDATE,
    review_text CLOB,
    source VARCHAR2(255),
    FOREIGN KEY (bank_id) REFERENCES banks(bank_id) ON DELETE CASCADE
);