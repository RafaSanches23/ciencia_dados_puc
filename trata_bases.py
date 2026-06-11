import os
import zipfile
import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURAÇÕES INICIAIS
# ============================================================

# Coloque o arquivo archive.zip na mesma pasta deste script.
# Caso o nome esteja diferente, altere abaixo.
ZIP_PATH = "archive.zip"

# Pasta onde os arquivos serão extraídos
EXTRACT_DIR = "dados_olist_extraidos"

# Pasta onde serão salvos os arquivos tratados
OUTPUT_DIR = "saida_olist_orange"

os.makedirs(EXTRACT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 2. EXTRAÇÃO DO ZIP
# ============================================================

with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)

print("Arquivos extraídos com sucesso:")
print(os.listdir(EXTRACT_DIR))


# ============================================================
# 3. LEITURA DOS ARQUIVOS CSV
# ============================================================

customers = pd.read_csv(os.path.join(EXTRACT_DIR, "olist_customers_dataset.csv"))
orders = pd.read_csv(os.path.join(EXTRACT_DIR, "olist_orders_dataset.csv"))
payments = pd.read_csv(os.path.join(EXTRACT_DIR, "olist_order_payments_dataset.csv"))
reviews = pd.read_csv(os.path.join(EXTRACT_DIR, "olist_order_reviews_dataset.csv"))
items = pd.read_csv(os.path.join(EXTRACT_DIR, "olist_order_items_dataset.csv"))
products = pd.read_csv(os.path.join(EXTRACT_DIR, "olist_products_dataset.csv"))
translation = pd.read_csv(os.path.join(EXTRACT_DIR, "product_category_name_translation.csv"))

print("\nDimensões originais:")
print("customers:", customers.shape)
print("orders:", orders.shape)
print("payments:", payments.shape)
print("reviews:", reviews.shape)
print("items:", items.shape)
print("products:", products.shape)
print("translation:", translation.shape)


# ============================================================
# 4. TRATAMENTO DAS DATAS
# ============================================================

date_columns = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date"
]

for col in date_columns:
    orders[col] = pd.to_datetime(orders[col], errors="coerce")


# ============================================================
# 5. AGREGAÇÃO DOS PAGAMENTOS
# ============================================================
# Alguns pedidos possuem mais de um pagamento.
# O objetivo é deixar apenas uma linha por pedido.

payment_type_by_value = (
    payments
    .sort_values(["order_id", "payment_value"], ascending=[True, False])
    .drop_duplicates("order_id")[["order_id", "payment_type"]]
    .rename(columns={"payment_type": "payment_type_principal"})
)

payments_agg = (
    payments
    .groupby("order_id", as_index=False)
    .agg(
        payment_value=("payment_value", "sum"),
        payment_installments=("payment_installments", "max"),
        payment_methods_qty=("payment_type", "nunique"),
        payment_transactions_qty=("payment_sequential", "max")
    )
    .merge(payment_type_by_value, on="order_id", how="left")
)

print("\nPagamentos agregados:", payments_agg.shape)


# ============================================================
# 6. TRATAMENTO DOS PRODUTOS E ITENS
# ============================================================
# Alguns pedidos possuem mais de um item.
# Aqui os valores são agregados por pedido.

products_treated = products.merge(
    translation,
    on="product_category_name",
    how="left"
)

products_treated["product_category_final"] = products_treated[
    "product_category_name_english"
].fillna(products_treated["product_category_name"])

products_treated["product_volume_cm3"] = (
    products_treated["product_length_cm"]
    * products_treated["product_height_cm"]
    * products_treated["product_width_cm"]
)

items_products = items.merge(
    products_treated[
        [
            "product_id",
            "product_category_final",
            "product_weight_g",
            "product_volume_cm3",
            "product_photos_qty"
        ]
    ],
    on="product_id",
    how="left"
)


def mode_or_nan(series):
    """
    Retorna a moda de uma coluna.
    Se não houver valor válido, retorna NaN.
    """
    mode_values = series.dropna().mode()
    if len(mode_values) == 0:
        return np.nan
    return mode_values.iloc[0]


items_agg = (
    items_products
    .groupby("order_id", as_index=False)
    .agg(
        total_items=("order_item_id", "count"),
        total_price=("price", "sum"),
        total_freight_value=("freight_value", "sum"),
        product_category=("product_category_final", mode_or_nan),
        product_weight_g=("product_weight_g", "mean"),
        product_volume_cm3=("product_volume_cm3", "mean"),
        product_photos_qty=("product_photos_qty", "mean")
    )
)

print("Itens agregados:", items_agg.shape)


# ============================================================
# 7. AGREGAÇÃO DAS AVALIAÇÕES
# ============================================================
# Alguns pedidos possuem mais de uma avaliação.
# A nota final será a média arredondada das avaliações do pedido.

reviews_agg = (
    reviews
    .groupby("order_id", as_index=False)
    .agg(
        review_score=("review_score", "mean")
    )
)

reviews_agg["review_score"] = reviews_agg["review_score"].round()

print("Avaliações agregadas:", reviews_agg.shape)


# ============================================================
# 8. JUNÇÃO DAS BASES
# ============================================================

df = (
    orders
    .merge(
        customers[["customer_id", "customer_city", "customer_state"]],
        on="customer_id",
        how="left"
    )
    .merge(
        payments_agg,
        on="order_id",
        how="left"
    )
    .merge(
        items_agg,
        on="order_id",
        how="left"
    )
    .merge(
        reviews_agg,
        on="order_id",
        how="left"
    )
)

print("\nBase consolidada antes dos filtros:", df.shape)


# ============================================================
# 9. FILTRO DOS PEDIDOS ENTREGUES
# ============================================================
# Para calcular prazo real de entrega, é necessário manter apenas pedidos entregues.

df = df[df["order_status"] == "delivered"].copy()

print("Base após manter apenas pedidos entregues:", df.shape)


# ============================================================
# 10. CRIAÇÃO DE VARIÁVEIS DERIVADAS
# ============================================================

df["purchase_month"] = df["order_purchase_timestamp"].dt.month
df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek

df["delivery_days"] = (
    df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
).dt.days

df["estimated_delivery_days"] = (
    df["order_estimated_delivery_date"] - df["order_purchase_timestamp"]
).dt.days

df["delay_days"] = (
    df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]
).dt.days

df["freight_ratio"] = (
    df["total_freight_value"] / (df["total_price"] + df["total_freight_value"])
)

df["average_item_price"] = df["total_price"] / df["total_items"]

# Coluna alvo categórica:
# review_score de 1 a 3 = baixa
# review_score de 4 a 5 = alta
df["review_class"] = np.where(df["review_score"] <= 3, "baixa", "alta")


# ============================================================
# 11. SELEÇÃO DAS COLUNAS FINAIS
# ============================================================
# Estas colunas serão usadas no Orange e na documentação.

final_columns = [
    "order_id",
    "customer_state",
    "customer_city",
    "purchase_month",
    "purchase_dayofweek",
    "payment_type_principal",
    "payment_installments",
    "payment_value",
    "payment_methods_qty",
    "total_items",
    "total_price",
    "total_freight_value",
    "freight_ratio",
    "average_item_price",
    "product_category",
    "product_weight_g",
    "product_volume_cm3",
    "product_photos_qty",
    "delivery_days",
    "estimated_delivery_days",
    "delay_days",
    "review_score",
    "review_class"
]

df_final = df[final_columns].copy()


# ============================================================
# 12. LIMPEZA FINAL
# ============================================================
# Remove infinitos e linhas com valores faltantes nas colunas selecionadas.
# Isso garante que as colunas finais estarão totalmente preenchidas.

df_final = df_final.replace([np.inf, -np.inf], np.nan)
df_final = df_final.dropna().copy()

# Ajustes de tipo
df_final["review_score"] = df_final["review_score"].astype(int)
df_final["purchase_month"] = df_final["purchase_month"].astype(int)
df_final["purchase_dayofweek"] = df_final["purchase_dayofweek"].astype(int)
df_final["payment_installments"] = df_final["payment_installments"].astype(int)
df_final["payment_methods_qty"] = df_final["payment_methods_qty"].astype(int)
df_final["total_items"] = df_final["total_items"].astype(int)
df_final["delivery_days"] = df_final["delivery_days"].astype(int)
df_final["estimated_delivery_days"] = df_final["estimated_delivery_days"].astype(int)
df_final["delay_days"] = df_final["delay_days"].astype(int)

# Arredondamento de colunas numéricas contínuas
float_columns = [
    "payment_value",
    "total_price",
    "total_freight_value",
    "freight_ratio",
    "average_item_price",
    "product_weight_g",
    "product_volume_cm3",
    "product_photos_qty"
]

for col in float_columns:
    df_final[col] = df_final[col].round(2)


print("\nBase final tratada:", df_final.shape)
print("\nValores faltantes na base final:")
print(df_final.isna().sum())

print("\nDistribuição da coluna alvo:")
print(df_final["review_class"].value_counts())
print(df_final["review_class"].value_counts(normalize=True).round(4))


# ============================================================
# 13. SALVAR CSV FINAL PARA O ORANGE
# ============================================================

output_csv = os.path.join(OUTPUT_DIR, "olist_financeiro_reviews_orange.csv")

df_final.to_csv(output_csv, index=False, encoding="utf-8-sig")

print("\nArquivo final salvo em:")
print(output_csv)


# ============================================================
# 14. GERAÇÃO DO RESUMO ESTATÍSTICO DAS COLUNAS
# ============================================================
# Este arquivo ajuda a preencher a tabela solicitada no DOCX.

tipo_sugerido = {
    "order_id": "Identificador",
    "customer_state": "Nominal",
    "customer_city": "Nominal",
    "purchase_month": "Ordinal/Numérico discreto",
    "purchase_dayofweek": "Ordinal/Numérico discreto",
    "payment_type_principal": "Nominal",
    "payment_installments": "Numérico discreto",
    "payment_value": "Numérico contínuo",
    "payment_methods_qty": "Numérico discreto",
    "total_items": "Numérico discreto",
    "total_price": "Numérico contínuo",
    "total_freight_value": "Numérico contínuo",
    "freight_ratio": "Numérico contínuo",
    "average_item_price": "Numérico contínuo",
    "product_category": "Nominal",
    "product_weight_g": "Numérico contínuo",
    "product_volume_cm3": "Numérico contínuo",
    "product_photos_qty": "Numérico discreto",
    "delivery_days": "Numérico discreto",
    "estimated_delivery_days": "Numérico discreto",
    "delay_days": "Numérico discreto",
    "review_score": "Ordinal/Numérico discreto",
    "review_class": "Nominal - coluna alvo"
}

descricao_colunas = {
    "order_id": "Identificador único do pedido.",
    "customer_state": "Estado do cliente.",
    "customer_city": "Cidade do cliente.",
    "purchase_month": "Mês em que o pedido foi realizado.",
    "purchase_dayofweek": "Dia da semana em que o pedido foi realizado, sendo 0 segunda-feira e 6 domingo.",
    "payment_type_principal": "Principal forma de pagamento usada no pedido.",
    "payment_installments": "Maior número de parcelas associado ao pedido.",
    "payment_value": "Valor total pago no pedido.",
    "payment_methods_qty": "Quantidade de métodos de pagamento diferentes usados no pedido.",
    "total_items": "Quantidade total de itens no pedido.",
    "total_price": "Soma dos preços dos itens do pedido.",
    "total_freight_value": "Soma dos valores de frete do pedido.",
    "freight_ratio": "Proporção do frete em relação ao valor total de produtos mais frete.",
    "average_item_price": "Preço médio dos itens do pedido.",
    "product_category": "Categoria predominante dos produtos do pedido.",
    "product_weight_g": "Peso médio dos produtos do pedido em gramas.",
    "product_volume_cm3": "Volume médio dos produtos do pedido em centímetros cúbicos.",
    "product_photos_qty": "Quantidade média de fotos dos produtos do pedido.",
    "delivery_days": "Quantidade de dias entre a compra e a entrega ao cliente.",
    "estimated_delivery_days": "Quantidade de dias entre a compra e a data estimada de entrega.",
    "delay_days": "Diferença, em dias, entre a entrega real e a entrega estimada.",
    "review_score": "Nota de avaliação dada pelo cliente, de 1 a 5.",
    "review_class": "Classe da avaliação: baixa para notas 1 a 3 e alta para notas 4 a 5."
}


def get_mode(series):
    mode_values = series.mode(dropna=True)
    if len(mode_values) == 0:
        return ""
    return mode_values.iloc[0]


def get_valid_values(series, is_numeric):
    if is_numeric:
        return f"Valores entre {series.min()} e {series.max()}"
    else:
        unique_values = series.dropna().astype(str).unique()
        sample_values = sorted(unique_values)[:10]
        if len(unique_values) > 10:
            return f"Categorias observadas, por exemplo: {sample_values} ... Total: {len(unique_values)} categorias"
        return f"Categorias observadas: {sample_values}"


summary_rows = []

for col in final_columns:
    series = df_final[col]
    is_numeric = pd.api.types.is_numeric_dtype(series)

    row = {
        "coluna": col,
        "representa": descricao_colunas.get(col, ""),
        "tipo_de_dado_sugerido": tipo_sugerido.get(col, ""),
        "valores_validos": get_valid_values(series, is_numeric),
        "valores_distintos": series.nunique(dropna=True),
        "menor_valor": series.min() if is_numeric else "",
        "maior_valor": series.max() if is_numeric else "",
        "moda": get_mode(series),
        "media": round(series.mean(), 4) if is_numeric else "",
        "desvio_padrao": round(series.std(), 4) if is_numeric else "",
        "mediana": round(series.median(), 4) if is_numeric else ""
    }

    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

output_summary = os.path.join(OUTPUT_DIR, "resumo_estatistico_colunas.csv")
summary_df.to_csv(output_summary, index=False, encoding="utf-8-sig")

print("\nResumo estatístico salvo em:")
print(output_summary)


# ============================================================
# 15. RELATÓRIO SIMPLES NO TERMINAL
# ============================================================

print("\nResumo geral para usar no trabalho:")
print(f"Linhas finais: {df_final.shape[0]}")
print(f"Colunas finais: {df_final.shape[1]}")
print("Coluna alvo: review_class")
print("Arquivo pronto para o Orange:", output_csv)

print("\nPrimeiras linhas da base final:")
print(df_final.head())