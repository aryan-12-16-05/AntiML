import pandas as pd

from src.pipeline.feature_engineering import add_advanced_historical_features, FEATURE_COLS


def test_advanced_historical_features_add_expected_columns():
    df = pd.DataFrame(
        [
            {
                "timestamp": "2024-01-01 00:00:00",
                "from_account": "A",
                "to_account": "B",
                "amount_received_usd": 100.0,
                "amount_paid_usd": 90.0,
                "is_laundering": 0,
                "is_self_transaction": 0,
                "same_bank": 1,
                "currency_changed": 0,
                "payment_format_enc": 1,
                "hour_of_day": 0,
                "day_of_week": 0,
                "is_weekend": 0,
                "from_entity_type_enc": 0,
                "to_entity_type_enc": 0,
                "from_risk_tier_enc": 0,
                "to_risk_tier_enc": 0,
            },
            {
                "timestamp": "2024-01-01 00:10:00",
                "from_account": "A",
                "to_account": "C",
                "amount_received_usd": 150.0,
                "amount_paid_usd": 120.0,
                "is_laundering": 1,
                "is_self_transaction": 0,
                "same_bank": 1,
                "currency_changed": 0,
                "payment_format_enc": 1,
                "hour_of_day": 1,
                "day_of_week": 1,
                "is_weekend": 0,
                "from_entity_type_enc": 0,
                "to_entity_type_enc": 0,
                "from_risk_tier_enc": 0,
                "to_risk_tier_enc": 0,
            },
        ]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    out = add_advanced_historical_features(df)

    required = {
        "pair_prior_tx_count",
        "pair_prior_received_amount",
        "source_prior_out_degree",
        "source_hop2_reachable_count",
        "source_hop3_reachable_count",
        "historical_2hop_path_count",
        "historical_3hop_path_count",
    }

    assert required.issubset(set(out.columns))
    assert "source_prior_tx_count" in out.columns
    assert all(col in FEATURE_COLS for col in required)
