-- Write your SQL query here
select
    e.name
    , e.salary
    , d.dept_name
FROM employees as e
    INNER JOIN departments as d
    ON e.dept_id = d.id
ORDER BY e.name;
