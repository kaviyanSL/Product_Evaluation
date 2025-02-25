create table usaer_prompt(
id INT AUTO_INCREMENT PRIMARY KEY,
prompt_text text,
prompt_keyword text,
user_name varchar(255),
insert_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);