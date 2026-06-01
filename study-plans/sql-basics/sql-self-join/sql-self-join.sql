-- Write your SQL query here
SELECT
    us1.username,
    COALESCE(us2.username, 'organic') AS referrer_name
FROM user_referrals AS us1
LEFT JOIN user_referrals AS us2 
    ON us1.referred_by = us2.id
ORDER BY 
    us1.username ASC;