CREATE TABLE raw_comments_amazon_meta (
    id INT AUTO_INCREMENT PRIMARY KEY,
    main_category text,
    title text,
    average_rating FLOAT,
    rating_number INT,
    features JSON,
    description JSON,
    price FLOAT,
    images JSON,
    videos JSON,
    store text,
    categories JSON,
    details JSON,
    parent_asin text,
    bought_together FLOAT,
    subtitle text,
    author text
);