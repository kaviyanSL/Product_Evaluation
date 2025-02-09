CREATE TABLE clustered_comment (
    id BIGINT  AUTO_INCREMENT PRIMARY KEY,         
    comment TEXT,                   
    cluster INT,                     
    insert_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP  
);