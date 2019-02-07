from decision_tree.attribute_entropy import AttributeEntropy

ATTRIBUTES = [
    ['good', 'high', 'good'],
    ['good', 'high', 'poor'],
    ['good', 'low', 'good'],
    ['good', 'low', 'poor'],
    ['average', 'high', 'good'],
    ['average', 'low', 'poor'],
    ['average', 'high', 'poor'],
    ['average', 'low', 'good'],
    ['low', 'high', 'good'],
    ['low', 'high', 'poor'],
    ['low', 'low', 'good'],
    ['low', 'low', 'poor'],
]

CLASSES = [
    'y',
    'y',
    'y',
    'n',
    'y',
    'n',
    'y',
    'n',
    'y',
    'n',
    'n',
    'n',
]

HIGH_INCOMES = [
    ['good', 'good'],
    ['good', 'poor'],
    ['average', 'good'],
    ['average', 'poor'],
    ['low', 'good'],
    ['low', 'poor'],
]

HI_CLASSES = [
    'y',
    'y',
    'y',
    'y',
    'y',
    'n',
]

FEATURE_NAMES = ['credit_score', 'income', 'collateral']

def test_finds_lowest_entropy_attribute():
    best = AttributeEntropy(ATTRIBUTES, CLASSES).lowest_attributes()
    assert(best == 1) #income

    best = AttributeEntropy(HIGH_INCOMES, CLASSES).lowest_attributes()
    assert(best == 1) #collatoral

def test_is_leaf():
    is_leaf = AttributeEntropy(['y'], ['y']).is_leaf(['y', 'y', 'y'])
    assert(True == is_leaf)

    is_leaf = AttributeEntropy(['y'], ['y']).is_leaf(['y', 'y', 'y', 'n'])
    assert(False == is_leaf)

