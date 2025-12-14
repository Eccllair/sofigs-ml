from typing import Any


class TreeNode:
    value: Any
    height: int
    childrens: list["TreeNode"]

    def __init__(self, value, height):
        self.value = value
        self.height = height
        self.childrens = []


class Tree:
    def __init__(self):
        self._root = TreeNode("", 0)

    @property
    def height(self):
        """
        высота дерева
        """
        return self._calculate_height(self._root)

    def _calculate_height(self, node: TreeNode):
        """
        метод подсчета высоты
        """
        if not node.childrens:
            return 1

        child_heights = [self._calculate_height(child) for child in node.childrens]
        return 1 + max(child_heights)

    def find_from(self, node: TreeNode, value: Any) -> TreeNode:
        """
        Поиск по значению
        """
        if node.value == value:
            return node

        for child in node.childrens:
            found = self.find_from(child, value)
            if found:
                return found

        raise ValueError(f"Родительский узел со значением {value} не найден")

    def append_to(self, node: TreeNode, value: Any) -> TreeNode:
        """
        Добавить новый узел элементу дерева
        """
        child_node = TreeNode(value, node.height + 1)
        node.childrens.append(child_node)
        return child_node

    def append_by(self, find_value: Any, value: Any) -> TreeNode:
        """
        Добавить новый узел по значению
        """
        return self.append_to(self.find_from(self._root, find_value), value)

    def _collect_levels(self, node: TreeNode, levels: list[list[Any]]):
        levels[node.height].append(node.value)

        for child in node.childrens:
            self._collect_levels(child, levels)

    def get_levels(self) -> list[list[Any]]:
        """
        Получить уповни дерева со списком элементов на каждом уровне
        """
        levels: list[list[Any]] = [[] for _ in range(self.height)]
        self._collect_levels(self._root, levels)
        return levels
