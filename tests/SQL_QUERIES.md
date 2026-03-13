1. Поиск сделок, где не сработал стоп-лосс (поиск багов) (то что описал в чек-листе)
- SELECT id, pair, close_profit 
- FROM trades 
- WHERE close_profit < -0.26;

2. Проверка, что бот сделал усреднение (DCA)
- SELECT trade_id, COUNT(*) AS buys
- FROM orders 
- WHERE side = 'buy' 
- GROUP BY trade_id 
- HAVING buys > 1;

3. Проверка текущего открытого баланса
- SELECT SUM(stake_amount)
- FROM trades
- WHERE is_open = 1;
