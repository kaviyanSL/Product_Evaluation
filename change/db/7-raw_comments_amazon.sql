CREATE TABLE raw_comments_amazon (
    id INT AUTO_INCREMENT PRIMARY KEY,
    rating INT,
    title VARCHAR(255),
    text TEXT,
    images TEXT,
    asin VARCHAR(255),
    parent_asin VARCHAR(255),
    user_id VARCHAR(255),
    timestamp DATETIME,
    helpful_vote INT,
    verified_purchase BOOLEAN
);