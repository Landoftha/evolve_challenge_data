-- Total clientes
SELECT COUNT(DISTINCT cliente_id) AS total_clientes FROM gold.vw_clientes_base;
-- Soma contratos
SELECT SUM(vl_total_contrato) AS soma_contratos FROM gold.vw_clientes_base;
-- NPS (válidas, promotores, detratores, NPS%)
WITH x AS (
  SELECT
    SUM(CASE WHEN nota_nps BETWEEN 9 AND 10 THEN 1 ELSE 0 END) AS promotores,
    SUM(CASE WHEN nota_nps BETWEEN 0 AND 6  THEN 1 ELSE 0 END) AS detratores,
    SUM(CASE WHEN nota_nps BETWEEN 0 AND 10 THEN 1 ELSE 0 END) AS validas
  FROM gold.vw_nps_respostas
)
SELECT
  promotores, detratores, validas,
  CAST(((promotores - detratores)*100.0)/NULLIF(validas,0) AS DECIMAL(5,2)) AS [NPS %]
FROM x;
-- NPS por mês
SELECT mes, promotores, detratores, total_respostas, nps_percentual
FROM gold.vw_nps_agregado_mensal
ORDER BY mes;




















