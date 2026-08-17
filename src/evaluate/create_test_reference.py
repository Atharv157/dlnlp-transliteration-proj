import json
import codecs
from xml.dom.minidom import Document

def create_test_xml(test_file, output_path):
    """Create test reference XML in NEWS format"""
    # Load test data
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = []
        for line in f:
            item = json.loads(line.strip())
            roman_word = item.get('english word', '').strip()
            devanagari_word = item.get('native word', '').strip()
            if roman_word and devanagari_word:
                test_data.append((roman_word, devanagari_word))
    
    doc = Document()
    
    # Create root element
    corpus = doc.createElement('TransliterationCorpus')
    corpus.setAttribute('SourceLang', 'Roman')
    corpus.setAttribute('TargetLang', 'Devanagari')
    corpus.setAttribute('CorpusID', 'Hindi_Test')
    corpus.setAttribute('CorpusType', 'test')
    corpus.setAttribute('CorpusSize', str(len(test_data)))
    corpus.setAttribute('CorpusFormat', 'NEWS')
    doc.appendChild(corpus)
    
    # Add test examples
    for i, (roman_word, devanagari_word) in enumerate(test_data):
        name_elem = doc.createElement('Name')
        name_elem.setAttribute('ID', str(i+1))
        
        # Source name
        src_elem = doc.createElement('SourceName')
        src_text = doc.createTextNode(roman_word)
        src_elem.appendChild(src_text)
        name_elem.appendChild(src_elem)
        
        # Target name (reference)
        tgt_elem = doc.createElement('TargetName')
        tgt_elem.setAttribute('ID', '1')  # Only one reference
        tgt_text = doc.createTextNode(devanagari_word)
        tgt_elem.appendChild(tgt_text)
        name_elem.appendChild(tgt_elem)
        
        corpus.appendChild(name_elem)
    
    # Write to file
    with codecs.open(output_path, 'w', 'utf-8') as f:
        f.write(doc.toprettyxml(indent='  '))
    
    print(f"✅ Test reference saved to: {output_path}")

def main():
    test_file = 'data/raw/hin/hin_test.json'
    output_xml = 'test_reference.xml'
    create_test_xml(test_file, output_xml)

if __name__ == "__main__":
    main()