
CREATE TABLE raw_comments_amazon_meta (
    id INT AUTO_INCREMENT PRIMARY KEY,
    main_category VARCHAR(255),
    title VARCHAR(255),
    average_rating FLOAT,
    rating_number INT,
    features TEXT,
    description TEXT,
    price FLOAT,
    images TEXT,
    videos TEXT,
    store VARCHAR(255),
    categories TEXT,
    details TEXT,
    parent_asin VARCHAR(255),
    bought_together FLOAT,
    subtitle VARCHAR(255),
    author VARCHAR(255)
);