-- =============================================================
-- MINIMAL GOLD VIEWS (1 fonte apenas)
-- Depende só de: stage.TB_STAGE_CLIENTES
-- =============================================================

-- cria schema se não existir
IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'gold')
    EXEC('CREATE SCHEMA gold');
GO

-- 1) Base de clientes
IF OBJECT_ID('gold.vw_clientes_base') IS NOT NULL
    DROP VIEW gold.vw_clientes_base;
GO
CREATE VIEW gold.vw_clientes_base AS
SELECT
    TRY_CAST(cliente_id AS BIGINT)                AS cliente_id,
    DS_PROD                                       AS produto,
    DS_LIN_REC                                    AS linha_receita,
    DS_SEGMENTO                                   AS segmento,
    DS_SUBSEGMENTO                                AS subsegmento,
    UF                                            AS uf,
    CIDADE                                        AS cidade,
    SITUACAO_CONTRATO                             AS situacao_contrato,
    TRY_CAST(VL_TOTAL_CONTRATO AS DECIMAL(18,2))  AS vl_total_contrato,
    TRY_CONVERT(date, DT_ASSINATURA_CONTRATO, 23) AS dt_assinatura_contrato
FROM stage.TB_STAGE_CLIENTES;
GO

-- 2) Respostas NPS
IF OBJECT_ID('gold.vw_nps_respostas') IS NOT NULL
    DROP VIEW gold.vw_nps_respostas;
GO
CREATE VIEW gold.vw_nps_respostas AS
SELECT
    TRY_CAST(cliente_id AS BIGINT)  AS cliente_id,
    DS_PROD                         AS produto,
    DS_SEGMENTO                     AS segmento,
    DS_SUBSEGMENTO                  AS subsegmento,
    TRY_CAST(resposta_NPS_x AS INT) AS nota_nps,
    CASE
        WHEN TRY_CAST(resposta_NPS_x AS INT) BETWEEN 0 AND 6 THEN 'detrator'
        WHEN TRY_CAST(resposta_NPS_x AS INT) IN (7,8) THEN 'neutro'
        WHEN TRY_CAST(resposta_NPS_x AS INT) IN (9,10) THEN 'promotor'
        ELSE NULL
    END AS grupo_nps,
    TRY_CONVERT(date, DT_ASSINATURA_CONTRATO, 23) AS dt_referencia
FROM stage.TB_STAGE_CLIENTES;
GO

-- 3) Agregado mensal de NPS
IF OBJECT_ID('gold.vw_nps_agregado_mensal') IS NOT NULL
    DROP VIEW gold.vw_nps_agregado_mensal;
GO
CREATE VIEW gold.vw_nps_agregado_mensal AS
WITH b AS (
    SELECT
        DS_PROD AS produto,
        DS_SEGMENTO AS segmento,
        EOMONTH(TRY_CONVERT(date, DT_ASSINATURA_CONTRATO, 23)) AS mes,
        CASE WHEN TRY_CAST(resposta_NPS_x AS INT) IN (9,10) THEN 1 ELSE 0 END AS is_promotor,
        CASE WHEN TRY_CAST(resposta_NPS_x AS INT) BETWEEN 0 AND 6 THEN 1 ELSE 0 END AS is_detrator
    FROM stage.TB_STAGE_CLIENTES
)
SELECT
    produto,
    segmento,
    mes,
    SUM(is_promotor) AS promotores,
    SUM(is_detrator) AS detratores,
    COUNT(*)         AS total_respostas,
    CAST( ( (SUM(is_promotor) - SUM(is_detrator)) * 100.0 ) / NULLIF(COUNT(*),0) AS DECIMAL(5,2) ) AS nps_percentual
FROM b
GROUP BY produto, segmento, mes;
GO
