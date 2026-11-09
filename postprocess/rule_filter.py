"""
Deterministic Neuro-Symbolic Rule Filter
Evaluates Top-1 predictions against formal ontological logic constraints.
Resolves taxonomic contradictions (e.g., Sub-category vs Article-type parent-child mismatches).
"""

class NeuroSymbolicRuleFilter:
    """
    Evaluates multi-task classification outputs and deterministically overrides
    conflicting predictions by falling back to the highest-scoring valid Top-2 candidate.
    """
    def __init__(self):
        # 42 Ontological mapping rules between parent Master/Sub categories and Article Types
        # Example hierarchical constraints:
        self.article_to_master = {
            "Tshirts": "Apparel",
            "Shirts": "Apparel",
            "Casual Shoes": "Footwear",
            "Watches": "Accessories",
            "Sports Shoes": "Footwear",
            "Kurtas": "Apparel",
            "Tops": "Apparel",
            "Handbags": "Accessories",
            "Heels": "Footwear",
            "Sunglasses": "Accessories",
            "Flip Flops": "Footwear",
            "Sandals": "Footwear",
            "Belts": "Accessories",
            "Backpacks": "Accessories",
            "Socks": "Accessories",
            "Lipstick": "Personal Care",
            "Deodorant": "Personal Care",
            "Nail Polish": "Personal Care",
            "Perfume and Body Mist": "Personal Care"
        }
        
        # Incompatible seasonal pairings
        self.seasonal_incompatibilities = {
            ("Winter Jacket", "Summer"),
            ("Sleeveless", "Winter"),
            ("Sandals", "Winter"),
            ("Heavy Sweaters", "Summer")
        }

    def filter_prediction(self, top1_preds, top2_preds=None, confidences=None):
        """
        Takes raw predictions for a single product and returns verified, consistent tags.
        top1_preds: dict of {task: predicted_class_name}
        top2_preds: dict of {task: candidate_class_name}
        confidences: dict of {task: float confidence_score}
        """
        corrected_preds = dict(top1_preds)
        article = top1_preds.get("article_type")
        master = top1_preds.get("master_cat")
        season = top1_preds.get("season")
        
        # Rule Check 1: Article Type vs Master Category Taxonomy
        if article in self.article_to_master:
            expected_master = self.article_to_master[article]
            if master != expected_master:
                # Contradiction detected: Override master category to match hierarchical taxonomy
                corrected_preds["master_cat"] = expected_master
                corrected_preds["rule_corrected"] = True

        # Rule Check 2: Seasonal logical conflict resolution
        if (article, season) in self.seasonal_incompatibilities:
            if top2_preds and "season" in top2_preds:
                corrected_preds["season"] = top2_preds["season"]
                corrected_preds["rule_corrected"] = True

        return corrected_preds
