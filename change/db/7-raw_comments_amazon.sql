CREATE TABLE raw_comments_amazon (
    id INT AUTO_INCREMENT PRIMARY KEY,
    rating INT,
    title TEXT,
    text TEXT,
    images json,
    asin TEXT,
    parent_asin TEXT,
    user_id TEXT,
    timestamp DATETIME,
    helpful_vote INT,
    verified_purchase BOOLEAN
);