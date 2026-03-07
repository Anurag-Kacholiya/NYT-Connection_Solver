import requests
import requests_cache

requests_cache.install_cache('wikidata_cache', expire_after=86400)

def get_wikidata_properties(word):
    url = "https://www.wikidata.org/w/api.php"
    
    # 1. Search for entity
    search_params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": word,
        "limit": 1
    }
    
    headers = {"User-Agent": "ConnectionsSolver/1.0 (test@example.com)"}
    
    try:
        search_res = requests.get(url, params=search_params, headers=headers).json()
        if not search_res.get('search'):
            return set()
            
        entity_id = search_res['search'][0]['id']
        
        # 2. Get properties for entity
        prop_params = {
            "action": "wbgetentities",
            "format": "json",
            "ids": entity_id,
            "props": "claims",
            "languages": "en"
        }
        
        prop_res = requests.get(url, params=prop_params, headers=headers).json()
        claims = prop_res.get('entities', {}).get(entity_id, {}).get('claims', {})
        
        # Extract interesting property values (as Q-nodes)
        # We can just extract all Q-node values from all claims to be broad, 
        # or restrict to specific properties. Let's extract ALL Q-node values for simplicity.
        properties = set()
        for prop_id, claim_list in claims.items():
            for claim in claim_list:
                try:
                    datavalue = claim['mainsnak']['datavalue']['value']
                    if isinstance(datavalue, dict) and datavalue.get('entity-type') == 'item':
                        properties.add(datavalue['id'])
                except KeyError:
                    pass
        
        return properties
    except Exception as e:
        print(f"Error for {word}: {e}")
        return set()

print("PUMP:", get_wikidata_properties("PUMP"))
print("BOOT:", get_wikidata_properties("BOOT"))
