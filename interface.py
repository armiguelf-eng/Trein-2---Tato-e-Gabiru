import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ==========================================
# 1. CARREGAMENTO DOS ARTEFATOS
# ==========================================
@st.cache_resource
def carregar_arquivos():
    # Como você salvou usando joblib.dump, precisamos ler com joblib.load
    # Certifique-se de que o .pkl está na mesma pasta que este script
    artefatos = joblib.load('modelo_credito_producao.pkl')
    return artefatos

# ==========================================
# 2. PIPELINE DE PREDIÇÃO COMPLETO
# ==========================================
def gerar_predicoes(df, art):
    d = df.copy()

    # --- A. FEATURE ENGINEERING BASE ---
    d["AGE_YEARS"] = (-d["DAYS_BIRTH"]) / 365
    d["YEARS_EMPLOYED"] = np.where(d["DAYS_EMPLOYED"] > 0, 0, -d["DAYS_EMPLOYED"] / 365)
    d["INCOME_LOG"] = np.log1p(d["AMT_INCOME_TOTAL"])
    d["INCOME_SQRT"] = np.sqrt(d["AMT_INCOME_TOTAL"])
    d["INCOME_PER_PERSON"] = d["AMT_INCOME_TOTAL"] / d["CNT_FAM_MEMBERS"].fillna(1).replace(0, 1)
    d["INCOME_PER_CHILD"] = d["AMT_INCOME_TOTAL"] / (d["CNT_CHILDREN"] + 1)
    d["INCOME_LOG_PP"] = np.log1p(d["INCOME_PER_PERSON"])
    d["INCOME_LOG_PC"] = np.log1p(d["INCOME_PER_CHILD"])
    
    ext = d[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
    d["EXT_MEAN"] = ext.mean(axis=1)
    d["EXT_MIN"] = ext.min(axis=1)
    d["EXT_2x3"] = d["EXT_SOURCE_2"] * d["EXT_SOURCE_3"]
    
    d["INCOME_SQ"] = d["INCOME_LOG"] ** 2
    d["INCOME_CB"] = d["INCOME_LOG"] ** 3
    d["INCOME_x_EXTMEAN"] = d["INCOME_LOG"] * d["EXT_MEAN"].fillna(0)
    d["INCOME_x_EMPLOYED"] = d["INCOME_LOG"] * d["YEARS_EMPLOYED"]
    d["INCOME_x_INCTPP"] = d["INCOME_LOG"] * d["INCOME_LOG_PP"]
    d["EMPLOYED_x_EXT2"] = d["YEARS_EMPLOYED"] * d["EXT_SOURCE_2"].fillna(0)
    d["EMPLOYED_RATIO"] = d["YEARS_EMPLOYED"] / (d["AGE_YEARS"] + 1e-5)
    
    d["INCOME_PCT"] = np.searchsorted(art['_income_sorted'], d["AMT_INCOME_TOTAL"].fillna(0).values) / art['_N_combined']
    d["EXT2_PCT"] = np.searchsorted(art['_ext2_sorted'], d["EXT_SOURCE_2"].fillna(0).values) / art['_N_combined']
    d["INCOME_PCT_x_EXT2"] = d["INCOME_PCT"] * d["EXT2_PCT"]
    
    d["FLAG_OWN_REALTY"] = d["FLAG_OWN_REALTY"].map({"Y": 1, "N": 0}).fillna(0)
    
    # Target Encoding
    for col in art['te_cols']:
        d[col + "_TE"] = d[col].map(art['te_maps'].get).fillna(art['gm'])
        d[col + "_TE_MED"] = d[col].map(art['te_maps_med'].get).fillna(art['gm'])
        d[col + "_TE_P75"] = d[col].map(art['te_maps_p75'].get).fillna(art['gm'])
        
    Xdf_tr = d[art['top30_features']]

    # --- B. TRANSFORMAÇÕES SCIKIT-LEARN ---
    X_tr = art['imp'].transform(Xdf_tr)
    X_tr_raw = Xdf_tr.values.astype(float)
    Xs_tr = art['sc'].transform(X_tr)
    Xsp_tr = art['sp'].transform(X_tr)
    Xsp_tr_s = art['sp_sc'].transform(Xsp_tr)
    XF_tr = np.hstack([Xs_tr, Xsp_tr_s])

    # --- C. CLUSTERIZAÇÃO (K-MEANS) ---
    X_core_tr = Xs_tr[:, art['core_idx']]
    cluster_tr = art['km'].predict(X_core_tr)
    dist_tr = art['km'].transform(X_core_tr)
    oh_tr = np.eye(art['km'].n_clusters)[cluster_tr]
    income_tr = Xs_tr[:, art['inc_idx']:art['inc_idx']+1]
    
    cluster_extra_tr = np.hstack([
        art['cluster_mean'][cluster_tr].reshape(-1, 1), 
        art['cluster_std'][cluster_tr].reshape(-1, 1), 
        dist_tr, oh_tr, oh_tr * income_tr
    ])
    
    cluster_extra_tr_s = art['ce_sc'].transform(cluster_extra_tr)
    XFC_tr = np.hstack([XF_tr, cluster_extra_tr_s])
    X_tree_tr = np.hstack([X_tr_raw, cluster_extra_tr])

    # --- D. PREDIÇÕES (ENSEMBLE) ---
    preds_ridge_kfold = np.mean([m.predict(XF_tr) for m in art['models_ridge']], axis=0)
    preds_cluster_kfold = np.mean([m.predict(XFC_tr) for m in art['models_cluster']], axis=0)
    preds_gbm_kfold = np.mean([m.predict(X_tree_tr) for m in art['models_gbm']], axis=0)

    w = art['weights']
    predicao_final_log = (w[0] * preds_ridge_kfold) + (w[1] * preds_cluster_kfold) + (w[2] * preds_gbm_kfold)
    
    return np.expm1(predicao_final_log)

# ==========================================
# 3. INTERFACE STREAMLIT
# ==========================================
def main():
    st.set_page_config(page_title="Inteligência de Crédito", page_icon="🏦", layout="wide")
    
    st.title("🏦 Motor de Decisão: Limite de Crédito")
    st.markdown("Bem-vindo ao painel analítico. Escolha a aba desejada abaixo:")

    try:
        artefatos = carregar_arquivos()
    except Exception as e:
        st.error(f"Erro ao carregar o modelo. Verifique se o arquivo .pkl está na pasta. Detalhes: {e}")
        return

    # Criando as Abas
    aba_lote, aba_simulador = st.tabs(["📂 Processamento em Lote (Planilha)", "🎛️ Simulador Individual"])

    # ==========================================
    # ABA 1: PROCESSAMENTO EM LOTE
    # ==========================================
    with aba_lote:
        arquivo_upload = st.file_uploader("Selecione a base de clientes (CSV ou Excel)", type=["csv", "xlsx"])

        if arquivo_upload is not None:
            with st.spinner('Processando modelo Ensemble. Aguarde...'):
                try:
                    df_original = pd.read_csv(arquivo_upload) if arquivo_upload.name.endswith('.csv') else pd.read_excel(arquivo_upload)
                    
                    if 'SK_ID_CURR' not in df_original.columns:
                        st.error("A planilha enviada não contém a coluna 'SK_ID_CURR'.")
                    else:
                        # Gerando predições e acoplando ao dataframe original para gráficos
                        predicoes_finais = gerar_predicoes(df_original, artefatos)
                        df_resultado = df_original.copy()
                        df_resultado['TARGET_CREDIT_LIMIT'] = predicoes_finais

                        st.success("Análise de crédito finalizada com sucesso!")

                        # --- KPIs ---
                        st.subheader("📊 Resumo Executivo")
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Clientes Analisados", f"{len(df_resultado):,}".replace(",", "."))
                        col2.metric("Limite Médio Aprovado", f"R$ {df_resultado['TARGET_CREDIT_LIMIT'].mean():,.2f}".replace(",", "_").replace(".", ",").replace("_", "."))
                        col3.metric("Risco Total de Exposição", f"R$ {df_resultado['TARGET_CREDIT_LIMIT'].sum():,.2f}".replace(",", "_").replace(".", ",").replace("_", "."))

                        st.divider()

                        # --- Filtro Interativo ---
                        st.subheader("🔍 Filtro Dinâmico")
                        limite_minimo = st.slider(
                            "Filtrar clientes com limite aprovado acima de:",
                            min_value=float(df_resultado['TARGET_CREDIT_LIMIT'].min()),
                            max_value=float(df_resultado['TARGET_CREDIT_LIMIT'].max()),
                            value=float(df_resultado['TARGET_CREDIT_LIMIT'].min()),
                            step=1000.0,
                            format="R$ %.2f"
                        )
                        df_filtrado = df_resultado[df_resultado['TARGET_CREDIT_LIMIT'] >= limite_minimo]

                        # --- Gráficos Interativos ---
                        col_grafico1, col_grafico2 = st.columns(2)
                        
                        with col_grafico1:
                            fig_hist = px.histogram(df_filtrado, x="TARGET_CREDIT_LIMIT", nbins=30, 
                                                    title="Distribuição dos Limites Aprovados",
                                                    labels={"TARGET_CREDIT_LIMIT": "Limite de Crédito", "count": "Quantidade de Clientes"},
                                                    color_discrete_sequence=['#1f77b4'])
                            st.plotly_chart(fig_hist, use_container_width=True)

                        with col_grafico2:
                            if 'NAME_INCOME_TYPE' in df_filtrado.columns:
                                df_agg = df_filtrado.groupby('NAME_INCOME_TYPE')['TARGET_CREDIT_LIMIT'].mean().reset_index()
                                fig_bar = px.bar(df_agg, x='NAME_INCOME_TYPE', y='TARGET_CREDIT_LIMIT',
                                                 title="Limite Médio por Tipo de Renda",
                                                 labels={"NAME_INCOME_TYPE": "Tipo de Renda", "TARGET_CREDIT_LIMIT": "Limite Médio"},
                                                 color='TARGET_CREDIT_LIMIT', color_continuous_scale='Blues')
                                st.plotly_chart(fig_bar, use_container_width=True)

                        # --- Tabela Final e Download ---
                        st.write(f"Mostrando **{len(df_filtrado)}** clientes após o filtro:")
                        st.dataframe(df_filtrado[['SK_ID_CURR', 'TARGET_CREDIT_LIMIT'] + [c for c in df_filtrado.columns if c not in ['SK_ID_CURR', 'TARGET_CREDIT_LIMIT']]].head(10))

                        csv_export = df_filtrado.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Baixar Base Filtrada (CSV)",
                            data=csv_export,
                            file_name='resultados_limite_credito.csv',
                            mime='text/csv'
                        )

                except Exception as e:
                    st.error(f"Ocorreu um erro durante o processamento: {e}")

    # ==========================================
    # ABA 2: SIMULADOR INDIVIDUAL
    # ==========================================
    with aba_simulador:
        st.subheader("🎛️ Simulação de Crédito em Tempo Real")
        st.write("Altere as variáveis abaixo para calcular o limite do cliente e avaliar uma transação específica.")

        with st.form("form_simulador"):
            st.markdown("### 👤 Dados do Cliente")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                idade = st.number_input("Idade (anos)", 18, 100, 35)
                renda = st.number_input("Renda Total (Anual/Mensal)", 0.0, 1000000.0, 50000.0, step=1000.0)
                membros_fam = st.number_input("Membros da Família", 1, 15, 2)
                tipo_renda = st.selectbox("Tipo de Renda", list(artefatos['te_maps']['NAME_INCOME_TYPE'].keys()))

            with col_b:
                anos_emprego = st.number_input("Anos de Emprego", 0.0, 50.0, 5.0, step=0.5)
                filhos = st.number_input("Quantidade de Filhos", 0, 10, 0)
                req_bureau = st.number_input("Consultas no Bureau de Crédito", 0, 20, 1)
                tipo_org = st.selectbox("Tipo de Organização", list(artefatos['te_maps']['ORGANIZATION_TYPE'].keys()))

            with col_c:
                ext_1 = st.slider("Score Externo 1", 0.0, 1.0, 0.5)
                ext_2 = st.slider("Score Externo 2", 0.0, 1.0, 0.5)
                ext_3 = st.slider("Score Externo 3", 0.0, 1.0, 0.5)
                flag_imovel = st.selectbox("Possui Imóvel Próprio?", ["Y", "N"])
            
            st.markdown("### 🛒 Avaliação da Transação")
            valor_transacao = st.number_input("Valor da Transação Desejada (R$)", min_value=0.0, max_value=1000000.0, value=5000.0, step=500.0)

            calcular = st.form_submit_button("Gerar Análise e Avaliar Transação", type="primary", use_container_width=True)

        if calcular:
            # Montando o dataframe de 1 linha
            cliente_dict = {
                "SK_ID_CURR": [999999],
                "DAYS_BIRTH": [-idade * 365],
                "DAYS_EMPLOYED": [-anos_emprego * 365],
                "AMT_INCOME_TOTAL": [renda],
                "CNT_FAM_MEMBERS": [membros_fam],
                "CNT_CHILDREN": [filhos],
                "EXT_SOURCE_1": [ext_1],
                "EXT_SOURCE_2": [ext_2],
                "EXT_SOURCE_3": [ext_3],
                "FLAG_OWN_REALTY": [flag_imovel],
                "NAME_INCOME_TYPE": [tipo_renda],
                "ORGANIZATION_TYPE": [tipo_org],
                "AMT_REQ_CREDIT_BUREAU_YEAR": [req_bureau]
            }
            df_cliente = pd.DataFrame(cliente_dict)

            try:
                limite_sugerido = gerar_predicoes(df_cliente, artefatos)[0]
                
                # Exibindo o Limite Aprovado
                st.metric(label="Limite Aprovado pelo Modelo", 
                          value=f"R$ {limite_sugerido:,.2f}".replace(",", "_").replace(".", ",").replace("_", "."))
                
                st.divider()
                st.subheader("🛑 Decisão da Transação")
                
                # Lógica de aprovação da transação
                if valor_transacao <= limite_sugerido:
                    valor_formatado = f"R$ {valor_transacao:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
                    st.success(f"✅ **TRANSAÇÃO APROVADA**: O valor da transação ({valor_formatado}) está contido no limite de crédito do cliente.")
                else:
                    valor_formatado = f"R$ {valor_transacao:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
                    st.error(f"❌ **TRANSAÇÃO NEGADA**: O valor da transação ({valor_formatado}) excede o limite de crédito aprovado.")
                    
            except Exception as e:
                st.error(f"Erro ao simular: Faltou tratar alguma coluna no modelo? Detalhe: {e}")

if __name__ == '__main__':
    main()