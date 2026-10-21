package handlers

import (
	"encoding/json"
	"reflect"
	"testing"
)

// fastinoAnswer is a real Fastino gliner2.5-base-v1 answer (spans trimmed) for all four tasks.
const fastinoAnswer = `{
 "order": [{"order_id": {"text": "1234"}, "price": {"text": "$99"}, "city": {"text": "Austin"}}],
 "entities": {"company": [{"text": "Apple", "confidence": 1.0}], "person": [{"text": "Tim Cook"}],
              "city": [{"text": "Cupertino"}, {"text": "Austin"}]},
 "sentiment": {"label": "positive", "confidence": 1.0},
 "relation_extraction": {"works_for": [{"head": {"text": "Tim Cook"}, "tail": {"text": "Apple"}}], "located_in": []}
}`

func TestDiffExtractionResults(t *testing.T) {
	// Local found one city less, one extra person, the same sentiment label in bare-string form,
	// no relation, and the same order record with different confidences.
	local := `{
	 "order": [{"order_id": {"text": "1234"}, "price": {"text": "$99"}, "city": {"text": "Austin"}}],
	 "entities": {"company": ["apple"], "person": ["Tim  Cook", "Investors"], "city": ["Cupertino"]},
	 "sentiment": "Positive",
	 "relation_extraction": {"works_for": []}
	}`
	d, err := DiffExtractionResults(json.RawMessage(fastinoAnswer), json.RawMessage(local))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]SetDiff{
		"company": {Both: []string{"apple"}, OnlyA: []string{}, OnlyB: []string{}},
		"person":  {Both: []string{"tim cook"}, OnlyA: []string{}, OnlyB: []string{"investors"}},
		"city":    {Both: []string{"cupertino"}, OnlyA: []string{"austin"}, OnlyB: []string{}},
	}
	if !reflect.DeepEqual(d.Entities, want) {
		t.Errorf("entities = %+v", d.Entities)
	}
	if c := d.Classifications["sentiment"]; !c.Same || c.A != "positive" {
		t.Errorf("sentiment = %+v", c)
	}
	if r := d.Relations["works_for"]; len(r.OnlyA) != 1 || r.OnlyA[0] != "tim cook -> apple" || len(r.Both) != 0 {
		t.Errorf("works_for = %+v", r)
	}
	if !d.OtherSame["order"] {
		t.Errorf("order should compare equal: %+v", d.OtherSame)
	}
	// matched: apple, tim cook, cupertino, sentiment = 4; total: 4 + austin + investors + the relation = 7
	if d.Agreement < 0.571 || d.Agreement > 0.572 {
		t.Errorf("agreement = %v, want 4/7", d.Agreement)
	}

	same, err := DiffExtractionResults(json.RawMessage(fastinoAnswer), json.RawMessage(fastinoAnswer))
	if err != nil || same.Agreement != 1 {
		t.Errorf("self diff agreement = %v (%v)", same.Agreement, err)
	}
}
