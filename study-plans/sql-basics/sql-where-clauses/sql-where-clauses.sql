-- Write your SQL query here
select
    name,
    salary
FROM employees
WHERE department IN ('Engineering', 'Marketing') AND salary > 70000;
