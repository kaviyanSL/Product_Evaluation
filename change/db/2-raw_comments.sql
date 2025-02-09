CREATE TABLE raw_comments (
    id BIGINT  AUTO_INCREMENT PRIMARY KEY,         
    comment TEXT,                                        
    insert_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP  
);