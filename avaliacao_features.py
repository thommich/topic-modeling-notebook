"""Piloto de qualidade e poder preditivo de features por cohort.

Esta fase usa registros individuais com target e features. Os identificadores
dos cohorts vêm da análise do cubo em ``analise_cohorts.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import sqrt
from statistics import NormalDist
from typing import Hashable, Sequence

import numpy as np
import pandas as pd

from analise_cohorts import ResultadoAnalise


@dataclass(frozen=True)
class CriteriosFeature:
    minimo_avaliavel: int = 200
    minimo_bons: int = 100
    minimo_maus: int = 30
    cobertura_minima: float = 0.70
    auc_ci_inferior_min: float = 0.52
    iv_minimo_categorica: float = 0.02
    quantis_iv: int = 5
    confianca: float = 0.95

    def __post_init__(self) -> None:
        if min(self.minimo_avaliavel, self.minimo_bons, self.minimo_maus) < 0:
            raise ValueError("Os mínimos de contagem não podem ser negativos.")
        if not 0 <= self.cobertura_minima <= 1:
            raise ValueError("cobertura_minima deve estar entre 0 e 1.")
        if not 0.5 <= self.auc_ci_inferior_min <= 1:
            raise ValueError("auc_ci_inferior_min deve estar entre 0,5 e 1.")
        if (self.iv_minimo_categorica < 0 or not isinstance(self.quantis_iv, int)
                or self.quantis_iv < 2 or not 0 < self.confianca < 1):
            raise ValueError("IV, quantis ou confiança inválidos.")


@dataclass
class ResultadoFeatures:
    feature_detail: pd.DataFrame
    feature_temporal: pd.DataFrame
    feature_summary: pd.DataFrame
    combination_feature_summary: pd.DataFrame


def _auc_e_ic(y: pd.Series, x: pd.Series, z: float) -> tuple[float, float, float]:
    valido = y.notna() & x.notna()
    alvo = y.loc[valido]
    valores = x.loc[valido]
    n_maus = int(alvo.eq(1).sum())
    n_bons = int(alvo.eq(0).sum())
    if n_maus == 0 or n_bons == 0 or valores.nunique() < 2:
        return np.nan, np.nan, np.nan
    postos = valores.rank(method="average")
    auc = float((postos[alvo.eq(1)].sum() - n_maus * (n_maus + 1) / 2)
                / (n_maus * n_bons))
    q1 = auc / (2 - auc) if auc < 2 else np.nan
    q2 = 2 * auc * auc / (1 + auc)
    variancia = (auc * (1 - auc) + (n_maus - 1) * (q1 - auc * auc)
                 + (n_bons - 1) * (q2 - auc * auc)) / (n_maus * n_bons)
    margem = z * sqrt(max(0.0, variancia))
    return auc, max(0.0, auc - margem), min(1.0, auc + margem)


def _iv_e_limite(y: pd.Series, codigos: pd.Series,
                 confianca: float) -> tuple[float, float]:
    validos = y.notna()
    tab = pd.crosstab(codigos.loc[validos], y.loc[validos]).reindex(
        columns=[0.0, 1.0], fill_value=0)
    if len(tab) < 2 or tab[0.0].sum() == 0 or tab[1.0].sum() == 0:
        return np.nan, np.nan
    bons = tab[0.0].to_numpy(dtype=float)
    maus = tab[1.0].to_numpy(dtype=float)
    pb = (bons + 0.5) / (bons.sum() + 0.5 * len(tab))
    pm = (maus + 0.5) / (maus.sum() + 0.5 * len(tab))
    iv = float(np.sum((pm - pb) * np.log(pm / pb)))
    # Limite inferior da distribuição posterior das proporções por classe.
    # Semente fixa torna o diagnóstico reproduzível sem reamostrar registros.
    rng = np.random.default_rng(1729)
    draws_bons = rng.dirichlet(bons + 0.5, size=200)
    draws_maus = rng.dirichlet(maus + 0.5, size=200)
    iv_draws = np.sum((draws_maus - draws_bons)
                      * np.log(draws_maus / draws_bons), axis=1)
    limite = float(np.quantile(iv_draws, (1 - confianca) / 2))
    return iv, limite


def _metricas(bloco: pd.DataFrame, feature: str, codigos: pd.Series,
              numerica: bool, criterios: CriteriosFeature, z: float) -> dict:
    x = bloco[feature]
    y = bloco["_mau"]
    aval = y.notna() & x.notna()
    n_bons = int(y.loc[aval].eq(0).sum())
    n_maus = int(y.loc[aval].eq(1).sum())
    n_avaliavel = n_bons + n_maus
    cobertura = float(x.notna().mean()) if len(x) else np.nan
    apta = (n_avaliavel >= criterios.minimo_avaliavel
            and n_bons >= criterios.minimo_bons
            and n_maus >= criterios.minimo_maus
            and cobertura >= criterios.cobertura_minima)
    auc, inf, sup = (_auc_e_ic(y, x, z) if numerica else (np.nan, np.nan, np.nan))
    iv, iv_limite = _iv_e_limite(y, codigos.loc[bloco.index], criterios.confianca)
    preditiva = (pd.notna(inf) and inf >= criterios.auc_ci_inferior_min) if numerica else (
        pd.notna(iv_limite) and iv_limite >= criterios.iv_minimo_categorica)
    return dict(n_total=len(bloco), n_performance=int(y.notna().sum()),
                n_feature=int(x.notna().sum()), n_avaliavel=n_avaliavel,
                n_bons=n_bons, n_maus=n_maus, cobertura=cobertura,
                taxa_missing=1 - cobertura, auc=auc, auc_ic_inf=inf,
                auc_ic_sup=sup, iv=iv, iv_limite_inf=iv_limite,
                amostra_suficiente=apta,
                passa_criterios=bool(apta and preditiva))


def avaliar_features(
    dados: pd.DataFrame,
    diagnostico: ResultadoAnalise,
    features: Sequence[Hashable],
    *,
    combinacoes: Sequence[int] | None = None,
    col_target: Hashable = "mau",
    col_safra: Hashable = "safra",
    col_amostra: Hashable = "amostra",
    amostra_desenvolvimento: object = "desenvolvimento",
    amostra_validacao: object = "validacao",
    criterios: CriteriosFeature = CriteriosFeature(),
) -> ResultadoFeatures:
    """Avalia features por cohort com bins/direção fixados no desenvolvimento.

    Se ``combinacoes`` for omitido, usa as três primeiras combinações viáveis
    de 2 ou 3 características do ranking do cubo. A base individual deve
    representar exatamente a mesma população do diagnóstico.
    """
    if not isinstance(dados, pd.DataFrame):
        raise TypeError("dados deve ser um DataFrame Pandas.")
    features = list(features)
    if not features or len(set(features)) != len(features):
        raise ValueError("Informe features únicas.")
    resumo = diagnostico.combination_summary
    detalhe = diagnostico.cohort_detail
    if combinacoes is None:
        elegiveis = resumo.loc[resumo["viavel"] & resumo["qtd_caracteristicas"].isin([2, 3])]
        combinacoes = elegiveis.head(3)["combination_id"].tolist()
    combinacoes = list(combinacoes)
    if not combinacoes or len(set(combinacoes)) != len(combinacoes):
        raise ValueError("Informe ao menos uma combinação única e viável.")
    if not set(combinacoes) <= set(resumo["combination_id"]):
        raise ValueError("Há combination_id desconhecido.")
    dimensoes = list(detalhe.columns[:detalhe.columns.get_loc("combination_id")])
    obrigatorias = set(dimensoes + features + [col_target, col_safra, col_amostra])
    faltantes = obrigatorias - set(dados.columns)
    if faltantes:
        raise ValueError(f"Colunas ausentes: {sorted(map(str, faltantes))}.")
    if set(features) & set(dimensoes + [col_target, col_safra, col_amostra]):
        raise ValueError("Features não podem ser dimensões, safra, amostra ou target.")
    if dados.index.has_duplicates:
        dados = dados.reset_index(drop=True)
    base = dados.copy()
    base["_mau"] = pd.to_numeric(base[col_target], errors="raise")
    if not base["_mau"].dropna().isin([0, 1]).all():
        raise ValueError("O target deve conter somente 0, 1 ou NaN.")
    amostras = set(base[col_amostra].dropna().unique())
    if amostra_desenvolvimento not in amostras or amostra_validacao not in amostras:
        raise ValueError("As amostras de desenvolvimento e validação devem existir.")
    if amostra_desenvolvimento == amostra_validacao:
        raise ValueError("Desenvolvimento e validação precisam ser distintos.")
    base = base.reset_index(drop=True)
    z = NormalDist().inv_cdf((1 + criterios.confianca) / 2)
    especificacoes = {}
    dev = base.loc[base[col_amostra].eq(amostra_desenvolvimento)]
    for feature in features:
        numerica = pd.api.types.is_numeric_dtype(base[feature])
        if numerica:
            base[feature] = pd.to_numeric(base[feature], errors="raise")
            auc_dev, _, _ = _auc_e_ic(dev["_mau"], base.loc[dev.index, feature], z)
            direcao = 1 if pd.isna(auc_dev) or auc_dev >= 0.5 else -1
            base[feature] = base[feature] * direcao
            validos = base.loc[dev.index, feature].dropna()
            bordas = np.unique(np.quantile(validos, np.linspace(0, 1, criterios.quantis_iv + 1))) if len(validos) else np.array([])
            cod = pd.Series(-1, index=base.index, dtype=int)
            if len(bordas) > 1:
                observado = base[feature].notna()
                cod.loc[observado] = np.searchsorted(bordas[1:-1],
                    base.loc[observado, feature].to_numpy(), side="right")
        else:
            direcao = None
            cod = pd.Series(pd.factorize(base[feature], sort=False)[0], index=base.index)
        especificacoes[feature] = (numerica, direcao, cod)

    linhas = []
    linhas_temporais = []
    for cid in combinacoes:
        r = resumo.loc[resumo["combination_id"].eq(cid)].iloc[0]
        cols = list(r["caracteristicas"])
        grupos = detalhe.loc[detalhe["combination_id"].eq(cid),
                             cols + ["grupo_id", "qtd_total", "qtd_performance", "qtd_maus"]]
        base_cohort = base.merge(grupos[cols + ["grupo_id"]], on=cols, how="left",
                                 validate="many_to_one", sort=False)
        if base_cohort["grupo_id"].isna().any():
            raise ValueError(f"Registros sem correspondência no cubo: {r['combinacao']}.")
        # O merge mantém a ordem original, necessária para localizar os bins.
        if len(base_cohort) != len(base) or not base_cohort.index.equals(base.index):
            raise ValueError("O vínculo de registros com os cohorts alterou a base.")
        conferido = base_cohort.groupby("grupo_id", dropna=False).agg(
            n_total=("_mau", "size"),
            n_perf=("_mau", "count"),
            n_maus=("_mau", lambda x: int(x.eq(1).sum())))
        esperado = grupos.set_index("grupo_id")
        for atual, cubo_col in [("n_total", "qtd_total"),
                                ("n_perf", "qtd_performance"),
                                ("n_maus", "qtd_maus")]:
            if not conferido[atual].reindex(esperado.index).eq(esperado[cubo_col]).all():
                raise ValueError(f"Base de features difere do cubo em {cubo_col} para {r['combinacao']}.")
        for (grupo_id, amostra), bloco in base_cohort.groupby(
                ["grupo_id", col_amostra], dropna=False, sort=False):
            comum = dict(combination_id=int(cid), combinacao=r["combinacao"],
                         grupo_id=int(grupo_id), amostra=amostra)
            for feature in features:
                numerica, direcao, cod = especificacoes[feature]
                linhas.append(dict(**comum, feature=feature,
                    tipo="numerica" if numerica else "categorica", direcao=direcao,
                    **_metricas(bloco, feature, cod, numerica, criterios, z)))
            for safra, periodo in bloco.groupby(col_safra, dropna=False, sort=False):
                for feature in features:
                    numerica, direcao, cod = especificacoes[feature]
                    linhas_temporais.append(dict(**comum, safra=safra, feature=feature,
                        tipo="numerica" if numerica else "categorica", direcao=direcao,
                        **_metricas(periodo, feature, cod, numerica, criterios, z)))
    feature_detail = pd.DataFrame(linhas)
    feature_temporal = pd.DataFrame(linhas_temporais)
    feature_detail["ranking_iv_cohort"] = feature_detail.groupby(
        ["combination_id", "grupo_id", "amostra"])["iv"].rank(
            ascending=False, method="min", na_option="bottom")

    resumos = []
    combinacao_resumos = []
    for cid in combinacoes:
        parte = feature_detail.loc[feature_detail["combination_id"].eq(cid)]
        val = parte.loc[parte["amostra"].eq(amostra_validacao)]
        tempo_val = feature_temporal.loc[
            feature_temporal["combination_id"].eq(cid)
            & feature_temporal["amostra"].eq(amostra_validacao)]
        n_grupos = int(detalhe["combination_id"].eq(cid).sum())
        for feature in features:
            v = val.loc[val["feature"].eq(feature)]
            dev_feature = parte.loc[parte["feature"].eq(feature)
                                    & parte["amostra"].eq(amostra_desenvolvimento)]
            tv = tempo_val.loc[tempo_val["feature"].eq(feature)]
            aucs = v["auc"].dropna()
            comparavel = v[["grupo_id", "auc", "iv"]].merge(
                dev_feature[["grupo_id", "auc", "iv"]], on="grupo_id",
                suffixes=("_val", "_dev"), validate="one_to_one")
            resumos.append(dict(combination_id=int(cid),
                combinacao=v["combinacao"].iloc[0], feature=feature,
                tipo=v["tipo"].iloc[0], n_grupos=n_grupos,
                n_grupos_aprovados=int(v["passa_criterios"].sum()),
                passa_todos_grupos=bool(len(v) == n_grupos and v["passa_criterios"].all()),
                cobertura_min=float(v["cobertura"].min()),
                cobertura_amplitude_pp=float((v["cobertura"].max() - v["cobertura"].min()) * 100),
                auc_min=float(aucs.min()) if not aucs.empty else np.nan,
                auc_amplitude=float(aucs.max() - aucs.min()) if not aucs.empty else np.nan,
                auc_ic_inf_min=float(v["auc_ic_inf"].min()) if v["auc_ic_inf"].notna().any() else np.nan,
                iv_min=float(v["iv"].min()) if v["iv"].notna().any() else np.nan,
                iv_limite_inf_min=float(v["iv_limite_inf"].min())
                    if v["iv_limite_inf"].notna().any() else np.nan,
                delta_auc_val_dev_medio=float((comparavel["auc_val"]
                    - comparavel["auc_dev"]).mean()) if comparavel["auc_val"].notna().any() else np.nan,
                delta_iv_val_dev_medio=float((comparavel["iv_val"]
                    - comparavel["iv_dev"]).mean()) if comparavel["iv_val"].notna().any() else np.nan,
                auc_dp_temporal=float(tv.groupby("grupo_id")["auc"].std().mean())
                    if tv["auc"].notna().any() else np.nan,
                iv_dp_temporal=float(tv.groupby("grupo_id")["iv"].std().mean())
                    if tv["iv"].notna().any() else np.nan,
                cobertura_amplitude_temporal_pp=float(
                    (tv.groupby("grupo_id")["cobertura"].max()
                     - tv.groupby("grupo_id")["cobertura"].min()).mean() * 100)))
        dev_rank = parte.loc[parte["amostra"].eq(amostra_desenvolvimento)].pivot(
            index="feature", columns="grupo_id", values="iv")
        correlacoes = []
        for a, b in combinations(dev_rank.columns, 2):
            pares = dev_rank[[a, b]].dropna()
            if len(pares) >= 2:
                corr = pares[a].rank().corr(pares[b].rank())
                if pd.notna(corr):
                    correlacoes.append(float(corr))
        combinacao_resumos.append(dict(combination_id=int(cid),
            combinacao=parte["combinacao"].iloc[0], n_grupos=n_grupos,
            n_features=len(features),
            correlacao_media_rankings_iv=float(np.mean(correlacoes)) if correlacoes else np.nan))
    feature_summary = pd.DataFrame(resumos)
    combination_feature_summary = pd.DataFrame(combinacao_resumos)
    agregados = feature_summary.groupby("combination_id").agg(
        n_features_aprovadas_todos=("passa_todos_grupos", "sum"),
        cobertura_amplitude_media_pp=("cobertura_amplitude_pp", "mean"),
        auc_amplitude_media=("auc_amplitude", "mean")).reset_index()
    combination_feature_summary = combination_feature_summary.merge(
        agregados, on="combination_id", validate="one_to_one")
    combination_feature_summary = combination_feature_summary.sort_values(
        ["n_features_aprovadas_todos", "correlacao_media_rankings_iv", "n_grupos"],
        ascending=[False, False, True], na_position="last").reset_index(drop=True)
    feature_summary = feature_summary.sort_values(
        ["combination_id", "passa_todos_grupos", "auc_ic_inf_min", "iv_min"],
        ascending=[True, False, False, False], na_position="last").reset_index(drop=True)
    return ResultadoFeatures(feature_detail, feature_temporal,
                             feature_summary, combination_feature_summary)
