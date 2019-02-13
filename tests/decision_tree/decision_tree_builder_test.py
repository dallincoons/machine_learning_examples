from decision_tree.decision_tree_builder import DecisionTreeBuilder

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

SIMPLE_ATTRIBUTES = [
    ['high', 'poor'],
    ['high', 'poor'],
    ['high', 'poor'],
    ['high', 'poor'],
    ['low', 'good'],
    ['low', 'good'],
    ['low', 'good'],
    ['low', 'good'],
]

SIMPLE_CLASSES = [
    'y',
    'y',
    'y',
    'n',
    'n',
    'n',
    'n',
    'n',
]

def test_creates_node_with_categories():
    tree = DecisionTreeBuilder().create(ATTRIBUTES, CLASSES, ['credit score', 'income', 'collateral'])
    print(tree)

data = {
    0 : 1,
    1: 2,

}

class BasicTree:
    def build(self, list):
        tree = {}
        # print(len(list))
        if len(list) == 0:
            return 'y'
        else:
            len2 = len(list)
            list.pop()
            subtree = self.build(list)
            tree[len2] = subtree
        return tree

def test_build_basic_tree():
    tree = BasicTree().build([1,2,3])
    # print(tree)
