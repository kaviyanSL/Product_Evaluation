create table classification_model (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(255),
    model_data LONGBLOB,
    insertdate TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
);