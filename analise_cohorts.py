"""Diagnóstico de segmentações a partir de um cubo Pandas já agregado.

O cubo deve conter todas as agregações (inclusive o total geral). As três
últimas colunas são, nesta ordem: maus, performance disponível e total.
Categorias ausentes de verdade precisam ser representadas por um valor
distinto do marcador usado pelo cubo para células totalizadoras.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from statistics import NormalDist
from typing import Hashable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CriteriosElegibilidade:
    minimo_total: int = 1000
    minimo_performance: int = 500
    minimo_bons: int = 100
    minimo_maus: int = 30
    cobertura_minima: float = 0.8

    def __post_init__(self) -> None:
        limites = (self.minimo_total, self.minimo_performance,
                   self.minimo_bons, self.minimo_maus)
        if any(not isinstance(v, int) or v < 0 for v in limites):
            raise ValueError("Os limites de contagem devem ser inteiros não negativos.")
        if not 0 <= self.cobertura_minima <= 1:
            raise ValueError("cobertura_minima deve estar entre 0 e 1.")


@dataclass
class ResultadoAnalise:
    combination_summary: pd.DataFrame
    cohort_detail: pd.DataFrame
    cohort_temporal: pd.DataFrame
    combination_comparison: pd.DataFrame


def _mascara_total(serie: pd.Series, marcador_total: object) -> pd.Series:
    if marcador_total is None or bool(pd.isna(marcador_total)):
        return serie.isna()
    return serie.eq(marcador_total).fillna(False)


def _intervalo_wilson(maus: pd.Series, performance: pd.Series,
                      confianca: float) -> tuple[pd.Series, pd.Series]:
    z = NormalDist().inv_cdf((1 + confianca) / 2)
    n = performance.astype(float)
    p = maus.div(n.where(n.ne(0)))
    divisor = 1 + z * z / n.where(n.ne(0))
    centro = (p + z * z / (2 * n.where(n.ne(0)))) / divisor
    margem = z * np.sqrt((p * (1 - p) + z * z / (4 * n.where(n.ne(0))))
                          / n.where(n.ne(0))) / divisor
    return (centro - margem).clip(lower=0), (centro + margem).clip(upper=1)


def _risco(grupos: pd.DataFrame) -> dict[str, float]:
    """Métricas sobre grupos elegíveis, ponderadas por performance conhecida."""
    validos = grupos.loc[grupos["elegivel"] & grupos["qtd_performance"].gt(0)]
    if len(validos) < 2:
        return dict(br_min=np.nan, br_max=np.nan, amplitude_br_pp=np.nan,
                    razao_risco=np.nan, desvio_br_ponderado=np.nan,
                    v_cramer=np.nan, iv_cohorts=np.nan)

    maus = validos["qtd_maus"].to_numpy(dtype=float)
    n = validos["qtd_performance"].to_numpy(dtype=float)
    bons = n - maus
    br = maus / n
    p = maus.sum() / n.sum()
    variancia = float(np.dot(n / n.sum(), (br - p) ** 2))
    br_min, br_max = float(br.min()), float(br.max())
    v = sqrt(variancia / (p * (1 - p))) if 0 < p < 1 else np.nan
    if maus.sum() > 0 and bons.sum() > 0:
        # Suavização apenas do IV, para grupos com zero bons ou maus.
        dist_maus = (maus + 0.5) / (maus.sum() + 0.5 * len(maus))
        dist_bons = (bons + 0.5) / (bons.sum() + 0.5 * len(bons))
        iv = float(np.sum((dist_maus - dist_bons) * np.log(dist_maus / dist_bons)))
    else:
        iv = np.nan
    return dict(br_min=br_min, br_max=br_max,
                amplitude_br_pp=100 * (br_max - br_min),
                razao_risco=br_max / br_min if br_min > 0 else np.nan,
                desvio_br_ponderado=sqrt(variancia), v_cramer=v,
                iv_cohorts=iv)


def _pareto(resumo: pd.DataFrame, com_tempo: bool) -> pd.Series:
    """Fronteira: maximiza V/cobertura/estabilidade, minimiza grupos."""
    colunas = ["v_cramer", "cobertura_elegivel", "qtd_grupos"]
    sinais = np.array([1, 1, -1], dtype=float)
    if com_tempo:
        colunas.append("persistencia_sinal")
        sinais = np.append(sinais, 1)
    valores = resumo[colunas].to_numpy(dtype=float) * sinais
    admissivel = resumo["viavel"].to_numpy() & np.isfinite(valores).all(axis=1)
    frente = np.zeros(len(resumo), dtype=bool)
    for i in np.flatnonzero(admissivel):
        outros = valores[admissivel]
        domina = np.all(outros >= valores[i], axis=1) & np.any(outros > valores[i], axis=1)
        frente[i] = not np.any(domina)
    return pd.Series(frente, index=resumo.index)


def analisar_cubo(
    cubo: pd.DataFrame,
    *,
    caracteristicas: Sequence[Hashable] | None = None,
    col_safra: Hashable | None = None,
    col_amostra: Hashable | None = None,
    marcador_total: object = None,
    criterios: CriteriosElegibilidade = CriteriosElegibilidade(),
    confianca: float = 0.95,
) -> ResultadoAnalise:
    """Produz ranking, detalhe, evolução temporal e comparações incrementais.

    As primeiras colunas do cubo são dimensões; as últimas três são contagens
    de maus, performance disponível (maus 0 ou 1) e registros totais. Uma
    dimensão totalizada recebe `marcador_total` (NaN/None por padrão).
    `col_safra` e `col_amostra`, quando presentes, são dimensões auxiliares
    excluídas das combinações. O cubo deve incluir os respectivos rollups.
    """
    if not isinstance(cubo, pd.DataFrame) or cubo.shape[1] < 4:
        raise ValueError("Forneça um DataFrame com dimensões e três colunas de contagem.")
    if not cubo.columns.is_unique:
        raise ValueError("Os nomes das colunas devem ser únicos.")
    if not 0 < confianca < 1:
        raise ValueError("confianca deve estar entre 0 e 1.")

    dimensoes = list(cubo.columns[:-3])
    metricas = list(cubo.columns[-3:])
    if col_safra is not None and col_safra not in dimensoes:
        raise ValueError("col_safra deve ser uma dimensão do cubo.")
    if col_amostra is not None and col_amostra not in dimensoes:
        raise ValueError("col_amostra deve ser uma dimensão do cubo.")
    auxiliares = {c for c in (col_safra, col_amostra) if c is not None}
    if len(auxiliares) != len([c for c in (col_safra, col_amostra) if c is not None]):
        raise ValueError("col_safra e col_amostra devem ser diferentes.")
    if caracteristicas is None:
        caracteristicas = [c for c in dimensoes if c not in auxiliares]
    else:
        caracteristicas = list(caracteristicas)
        if len(set(caracteristicas)) != len(caracteristicas) or not set(caracteristicas) <= set(dimensoes) - auxiliares:
            raise ValueError("caracteristicas deve conter dimensões únicas, sem safra/amostra.")
        if set(caracteristicas) != set(dimensoes) - auxiliares:
            raise ValueError("Todas as dimensões não auxiliares devem estar em caracteristicas.")
    if not caracteristicas:
        raise ValueError("O cubo precisa de pelo menos uma característica.")

    dados = cubo.copy()
    for col in metricas:
        dados[col] = pd.to_numeric(dados[col], errors="raise")
        if dados[col].isna().any() or not np.isfinite(dados[col]).all():
            raise ValueError(f"{col}: contagens precisam ser finitas e sem nulos.")
        if (dados[col] < 0).any() or (dados[col] % 1 != 0).any():
            raise ValueError(f"{col}: contagens precisam ser inteiras não negativas.")
    maus, perf, total = metricas
    if (dados[maus] > dados[perf]).any() or (dados[perf] > dados[total]).any():
        raise ValueError("Cada célula deve satisfazer 0 <= maus <= performance <= total.")

    flags = pd.DataFrame({c: ~_mascara_total(dados[c], marcador_total)
                          for c in dimensoes}, index=dados.index)
    pesos = {c: 1 << i for i, c in enumerate(dimensoes)}
    dados["_mascara_cubo"] = sum(flags[c].astype(int) * pesos[c] for c in dimensoes)
    if dados.duplicated(subset=dimensoes).any():
        raise ValueError("Há células duplicadas no cubo; use um marcador de totalização distinto de categorias reais ausentes.")
    geral = dados.loc[dados["_mascara_cubo"].eq(0)]
    if len(geral) != 1 or geral.iloc[0][total] <= 0:
        raise ValueError("O cubo precisa de exatamente uma célula de total geral positiva.")
    n_geral = int(geral.iloc[0][total])
    br_geral = (float(geral.iloc[0][maus] / geral.iloc[0][perf])
                if geral.iloc[0][perf] > 0 else np.nan)
    for auxiliar in auxiliares:
        rollup = dados.loc[dados["_mascara_cubo"].eq(pesos[auxiliar]), metricas]
        if (rollup.empty or any(rollup[col].sum() != geral.iloc[0][col]
                                for col in metricas)):
            raise ValueError(f"Rollup de {auxiliar} incompleto ou inconsistente.")

    resumo_linhas: list[dict] = []
    detalhes: list[pd.DataFrame] = []
    temporais: list[pd.DataFrame] = []
    mascara_safra = pesos[col_safra] if col_safra is not None else 0
    mascara_amostra = pesos[col_amostra] if col_amostra is not None else 0

    for k in range(1, len(caracteristicas) + 1):
        for cols in combinations(caracteristicas, k):
            combinacao_id = sum(1 << caracteristicas.index(c) for c in cols)
            mascara = sum(pesos[c] for c in cols)
            grupo = dados.loc[dados["_mascara_cubo"].eq(mascara) & dados[total].gt(0)].copy()
            if (grupo.empty or grupo[total].sum() != n_geral
                    or grupo[perf].sum() != geral.iloc[0][perf]
                    or grupo[maus].sum() != geral.iloc[0][maus]):
                raise ValueError(f"Partição incompleta ou inconsistente: {cols}.")
            grupo = grupo.rename(columns={maus: "qtd_maus", perf: "qtd_performance", total: "qtd_total"})
            grupo["qtd_bons"] = grupo["qtd_performance"] - grupo["qtd_maus"]
            grupo["qtd_sem_performance"] = grupo["qtd_total"] - grupo["qtd_performance"]
            grupo["bad_rate"] = grupo["qtd_maus"].div(grupo["qtd_performance"].replace(0, np.nan))
            grupo["participacao"] = grupo["qtd_total"] / n_geral
            grupo["br_relativo_pp"] = 100 * (grupo["bad_rate"] - br_geral)
            grupo["elegivel"] = (
                grupo["qtd_total"].ge(criterios.minimo_total)
                & grupo["qtd_performance"].ge(max(1, criterios.minimo_performance))
                & grupo["qtd_bons"].ge(criterios.minimo_bons)
                & grupo["qtd_maus"].ge(criterios.minimo_maus)
            )
            grupo["ic_br_inf"], grupo["ic_br_sup"] = _intervalo_wilson(
                grupo["qtd_maus"], grupo["qtd_performance"], confianca)
            grupo["combination_id"] = combinacao_id
            grupo["combinacao"] = "+".join(map(str, cols))
            grupo["grupo_id"] = np.arange(1, len(grupo) + 1)
            detalhes.append(grupo[list(caracteristicas) + [
                "combination_id", "combinacao", "grupo_id", "qtd_total",
                "qtd_performance", "qtd_sem_performance", "qtd_bons", "qtd_maus",
                "participacao", "bad_rate", "br_relativo_pp", "ic_br_inf",
                "ic_br_sup", "elegivel"]])

            elegiveis = grupo.loc[grupo["elegivel"]]
            risco = _risco(grupo)
            linha = dict(combination_id=combinacao_id, combinacao="+".join(map(str, cols)),
                         caracteristicas=tuple(cols), qtd_caracteristicas=k,
                         qtd_grupos=len(grupo), qtd_grupos_elegiveis=len(elegiveis),
                         pct_grupos_elegiveis=len(elegiveis) / len(grupo),
                         menor_grupo_n=int(grupo["qtd_total"].min()),
                         menor_grupo_pct=float(grupo["participacao"].min()),
                         menor_qtd_maus=int(grupo["qtd_maus"].min()),
                         cobertura_elegivel=float(elegiveis["qtd_total"].sum() / n_geral),
                         br_geral=br_geral, **risco)
            linha["viavel_global"] = (linha["qtd_grupos_elegiveis"] >= 2
                                      and linha["cobertura_elegivel"] >= criterios.cobertura_minima)
            linha["viavel"] = linha["viavel_global"]
            if col_amostra is not None:
                amostras = dados.loc[dados["_mascara_cubo"].eq(mascara_amostra),
                                      [col_amostra, maus, perf, total]]
                grupos_amostra = dados.loc[dados["_mascara_cubo"].eq(mascara | mascara_amostra)
                                           & dados[total].gt(0)].copy()
                if amostras.empty or grupos_amostra.empty:
                    raise ValueError(f"Faltam células por amostra para {cols}.")
                valores_v, coberturas, viabilidades = [], [], []
                for valor_amostra, base_amostra in amostras.groupby(col_amostra, dropna=False):
                    # Comparação explícita preserva também uma categoria real NaN.
                    if pd.isna(valor_amostra):
                        chave = grupos_amostra[col_amostra].isna()
                    else:
                        chave = grupos_amostra[col_amostra].eq(valor_amostra)
                    bloco = grupos_amostra.loc[chave].rename(
                        columns={maus: "qtd_maus", perf: "qtd_performance", total: "qtd_total"}).copy()
                    esperado = base_amostra.iloc[0]
                    if (bloco.empty or bloco["qtd_total"].sum() != esperado[total]
                            or bloco["qtd_performance"].sum() != esperado[perf]
                            or bloco["qtd_maus"].sum() != esperado[maus]):
                        raise ValueError(f"Células por amostra inconsistentes para {cols}.")
                    bloco["qtd_bons"] = bloco["qtd_performance"] - bloco["qtd_maus"]
                    bloco["elegivel"] = (
                        bloco["qtd_total"].ge(criterios.minimo_total)
                        & bloco["qtd_performance"].ge(max(1, criterios.minimo_performance))
                        & bloco["qtd_bons"].ge(criterios.minimo_bons)
                        & bloco["qtd_maus"].ge(criterios.minimo_maus))
                    valores_v.append(_risco(bloco)["v_cramer"])
                    coberturas.append(float(bloco.loc[bloco["elegivel"], "qtd_total"].sum()
                                            / esperado[total]))
                    viabilidades.append(bloco["elegivel"].sum() >= 2
                                         and coberturas[-1] >= criterios.cobertura_minima)
                finitos = [v for v in valores_v if np.isfinite(v)]
                linha["qtd_amostras_avaliadas"] = len(valores_v)
                linha["v_min_amostras"] = min(finitos) if finitos else np.nan
                linha["v_max_amostras"] = max(finitos) if finitos else np.nan
                linha["cobertura_min_amostras"] = min(coberturas)
                linha["viavel_amostras"] = all(viabilidades)
                linha["viavel"] = linha["viavel_global"] and linha["viavel_amostras"]
            resumo_linhas.append(linha)

            if col_safra is not None:
                tempo = dados.loc[dados["_mascara_cubo"].eq(mascara | mascara_safra)
                                  & dados[total].gt(0)].copy()
                if tempo.empty:
                    raise ValueError(f"Faltam células por safra para {cols}.")
                tempo = tempo.rename(columns={maus: "qtd_maus", perf: "qtd_performance", total: "qtd_total"})
                tempo["qtd_bons"] = tempo["qtd_performance"] - tempo["qtd_maus"]
                tempo["bad_rate"] = tempo["qtd_maus"].div(tempo["qtd_performance"].replace(0, np.nan))
                safras = dados.loc[dados["_mascara_cubo"].eq(mascara_safra),
                                     [col_safra, maus, perf, total]].rename(
                                         columns={maus: "maus_safra", perf: "perf_safra", total: "total_safra"})
                tempo = tempo.merge(safras, on=col_safra, how="left", validate="many_to_one")
                if tempo["total_safra"].isna().any():
                    raise ValueError("Faltam totais de uma ou mais safras no cubo.")
                conferido = tempo.groupby(col_safra, dropna=False)[
                    ["qtd_maus", "qtd_performance", "qtd_total"]].sum().reset_index()
                conferido = conferido.merge(safras, on=col_safra, how="outer",
                                            validate="one_to_one", indicator=True)
                if (conferido["_merge"].ne("both").any()
                        or conferido["qtd_maus"].ne(conferido["maus_safra"]).any()
                        or conferido["qtd_performance"].ne(conferido["perf_safra"]).any()
                        or conferido["qtd_total"].ne(conferido["total_safra"]).any()):
                    raise ValueError(f"Células por safra inconsistentes para {cols}.")
                tempo["participacao_safra"] = tempo["qtd_total"] / tempo["total_safra"]
                tempo["br_safra"] = tempo["maus_safra"].div(tempo["perf_safra"].replace(0, np.nan))
                tempo["br_relativo_safra_pp"] = 100 * (tempo["bad_rate"] - tempo["br_safra"])
                tempo = tempo.merge(grupo[list(cols) + ["grupo_id", "elegivel"]],
                                    on=list(cols), how="left", validate="many_to_one")
                if tempo["grupo_id"].isna().any():
                    raise ValueError("Grupo por safra ausente do detalhe geral.")
                tempo["combination_id"] = combinacao_id
                tempo["combinacao"] = "+".join(map(str, cols))
                temporais.append(tempo[list(caracteristicas) + [col_safra,
                    "combination_id", "combinacao", "grupo_id", "qtd_total",
                    "qtd_performance", "qtd_bons", "qtd_maus", "participacao_safra",
                    "bad_rate", "br_safra", "br_relativo_safra_pp", "elegivel"]])

    resumo = pd.DataFrame(resumo_linhas)
    detalhe = pd.concat(detalhes, ignore_index=True)
    if temporais:
        temporal = pd.concat(temporais, ignore_index=True)
        for combinacao_id, bloco in temporal.groupby("combination_id", sort=False):
            linha_idx = resumo.index[resumo["combination_id"].eq(combinacao_id)][0]
            base = detalhe.loc[detalhe["combination_id"].eq(combinacao_id),
                               ["grupo_id", "participacao", "br_relativo_pp", "elegivel"]]
            todos = bloco.merge(base, on="grupo_id", suffixes=("", "_geral"))
            elegiveis_base = base.loc[base["elegivel"]].set_index("grupo_id")
            if elegiveis_base.empty:
                continue
            safras_observadas = dados.loc[dados["_mascara_cubo"].eq(mascara_safra), col_safra].tolist()
            participacoes = todos.loc[todos["elegivel"]].pivot_table(
                index="grupo_id", columns=col_safra, values="participacao_safra",
                aggfunc="first").reindex(index=elegiveis_base.index,
                                         columns=safras_observadas).fillna(0)
            dp_participacao = participacoes.std(axis=1).fillna(0)
            resumo.loc[linha_idx, "instabilidade_participacao_pp"] = float(
                np.average(dp_participacao, weights=elegiveis_base["participacao"]) * 100)
            validos = todos.loc[todos["elegivel"] & todos["bad_rate"].notna()]
            if validos.empty:
                continue
            dp_br = validos.groupby("grupo_id")["br_relativo_safra_pp"].std()
            dp_br = dp_br.reindex(elegiveis_base.index).fillna(0)
            resumo.loc[linha_idx, "instabilidade_br_relativo_pp"] = float(
                np.average(dp_br, weights=elegiveis_base["participacao"]))
            referencia = np.sign(validos["br_relativo_pp"])
            mesmo_sinal = np.sign(validos["br_relativo_safra_pp"]).eq(referencia)
            resumo.loc[linha_idx, "persistencia_sinal"] = float(np.average(
                mesmo_sinal.astype(float), weights=validos["qtd_performance"]))
            # Correlação de Spearman entre safras consecutivas nos grupos comuns.
            matriz = validos.pivot_table(index="grupo_id", columns=col_safra,
                                         values="br_relativo_safra_pp", aggfunc="first")
            correlacoes = []
            for a, b in zip(matriz.columns[:-1], matriz.columns[1:]):
                pares = matriz[[a, b]].dropna()
                if len(pares) >= 2:
                    cor = pares[a].rank().corr(pares[b].rank())
                    if pd.notna(cor):
                        correlacoes.append(cor)
            resumo.loc[linha_idx, "correlacao_ordem_safras"] = (
                float(np.mean(correlacoes)) if correlacoes else np.nan)
    else:
        temporal = pd.DataFrame(columns=list(caracteristicas) + ["combination_id", "combinacao",
            "grupo_id", "qtd_total", "qtd_performance", "qtd_bons", "qtd_maus",
            "participacao_safra", "bad_rate", "br_safra", "br_relativo_safra_pp", "elegivel"])
    for col in ("instabilidade_participacao_pp", "instabilidade_br_relativo_pp",
                "persistencia_sinal", "correlacao_ordem_safras"):
        if col not in resumo:
            resumo[col] = np.nan

    por_id = resumo.set_index("combination_id")
    comparacoes = []
    for linha in resumo.itertuples(index=False):
        if linha.qtd_caracteristicas < 2:
            continue
        for anterior in combinations(linha.caracteristicas, linha.qtd_caracteristicas - 1):
            anterior_id = sum(1 << caracteristicas.index(c) for c in anterior)
            base = por_id.loc[anterior_id]
            comparacoes.append(dict(combination_id=linha.combination_id,
                combinacao=linha.combinacao, subset_id=anterior_id,
                subset=base["combinacao"], caracteristica_adicionada=next(
                    c for c in linha.caracteristicas if c not in anterior),
                delta_v_cramer=linha.v_cramer - base["v_cramer"],
                delta_cobertura=linha.cobertura_elegivel - base["cobertura_elegivel"],
                grupos_adicionais=linha.qtd_grupos - base["qtd_grupos"]))
    comparacao = pd.DataFrame(comparacoes)
    resumo["ganho_v_melhor_subset"] = np.nan
    resumo["melhor_subset_id"] = pd.Series(pd.NA, index=resumo.index, dtype="Int64")
    if not comparacao.empty:
        for cid, bloco in comparacao.groupby("combination_id"):
            base_v = por_id.loc[bloco["subset_id"], "v_cramer"].to_numpy(dtype=float)
            if np.isfinite(base_v).any():
                melhor = bloco.iloc[int(np.nanargmax(base_v))]
                idx = resumo.index[resumo["combination_id"].eq(cid)][0]
                resumo.loc[idx, "melhor_subset_id"] = int(melhor["subset_id"])
                resumo.loc[idx, "ganho_v_melhor_subset"] = melhor["delta_v_cramer"]
    resumo["fronteira_pareto"] = _pareto(resumo, col_safra is not None)
    resumo = resumo.sort_values(["viavel", "v_cramer", "cobertura_elegivel",
                                 "qtd_caracteristicas", "qtd_grupos"],
                                ascending=[False, False, False, True, True],
                                na_position="last").reset_index(drop=True)
    resumo.insert(0, "ranking", np.arange(1, len(resumo) + 1))
    return ResultadoAnalise(resumo, detalhe, temporal, comparacao)
