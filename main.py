from tkinter import *
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import os
import sys
from collections import deque
import math


class GraphApp:
    def __init__(self):
        self.window = Tk()
        self.window.resizable(True, True)
        self.window.geometry('800x720+470+130')
        self.window.title('DFS/BFS')
        self.window.config(bg='#1a1a1a')

        self.text_1 = Text(self.window, font=('Arial', 15, 'bold'))
        self.text_1.tag_configure("center", justify='center')
        self.text_1.place(x=10, y=10, width=150, height=100)

        self.output_frame = Text(self.window, bg='white', foreground='black')
        self.output_frame.place(x=10, y=120, width=600, height=500)

        self.text_2 = Text(self.window, font=('Arial', 13, 'bold'),foreground='black',bg='white')
        self.text_2.tag_configure("center", justify='center')
        self.text_2.place(x=10, y=620, width=400, height=80)

        self.btn_1 = Button(self.window, text='Построить граф', command=self.display_graph)
        self.btn_1.place(x=210, y=10, width=100, height=50)

        self.btn_2 = Button(self.window, text='Очистить', command=self.restart_program)
        self.btn_2.place(x=320, y=10, width=100, height=50)

        self.btn_3 = Button(self.window, text='Обойти в глубину', command=self.dfs)
        self.btn_3.place(x=430, y=10, width=100, height=50)

        self.btn_4 = Button(self.window, text='Обойти в ширину', command=self.bfs)
        self.btn_4.place(x=430, y=70, width=100, height=50)

        self.text_3 = Text(self.window, font=('Arial', 13, 'bold'))
        self.text_3.tag_configure("center", justify='center')
        self.text_3.place(x=320, y=70, width=100, height=50)

        self.btn_5 = Button(self.window, text='Ввод вершины', command=self.vertex)
        self.btn_5.place(x=320, y=10, width=100, height=50)

        self.btn_6 = Button(self.window, text='Дейкстры', command=self.dijxtra)
        self.btn_6.place(x=540, y=10, width=100, height=50)

        self.btn_7 = Button(self.window, text='Флойда', command=self.floyd)
        self.btn_7.place(x=540, y=70, width=100, height=50)

        """  self.text_4 = Text(self.window, font=('Arial', 13, 'bold'))
        self.text_4.tag_configure("center", justify='center')
        self.text_4.place(x=10, y=620, width=400, height=80) """

        self.e1 = Entry(self.window)#Text(self.window, font=('Arial', 15, 'bold'))
        self.e1.insert(0, '0,1')
        self.e1.place(x=650, y=10, width=100, height=50)
        self.vert = 0
        self.start=0
        self.end = 1


    def vertex(self):
        self.vert = int(self.text_3.get('1.0', 'end-1c').replace(' ', ''))

    def run(self):
        self.window.mainloop()

    def display_graph(self):
        try: 
            self.canvas.get_tk_widget().pack_forget()
        except AttributeError: 
            pass                
        self.output_frame.delete('1.0', END)
        text = "0,3,2,10\n3,0,4,0\n2,4,0,2\n10,0,2,0"#self.text_1.get('1.0', 'end-1c').replace(' ', '')
        self.adjacency_matrix=[]
        for i in text.split('\n'):
            row=[]
            for j in i.split(','):
                row.append(int(j))
            self.adjacency_matrix.append(row)
        #self.adjacency_matrix = np.matrix([[int(j) for j in i] for i in text.split('\n')])
        self.nparr= np.array(self.adjacency_matrix)
        # Создим сетевой график
        self.G = nx.from_numpy_array(self.nparr, create_using=nx.DiGraph())

        # Создим фигуру и нарисуем на ней график
        fig = plt.figure(figsize=(2, 2))
        labels = nx.get_edge_attributes(self.G, "weight")
        pos = nx.spring_layout(self.G)
        nx.draw(self.G, pos,with_labels=True,  node_color='lightblue', node_size=200, font_size=12, font_weight='bold')
        edges = [(u, v) for (u, v, d) in self.G.edges(data=True) ]
        nx.draw_networkx_edges(self.G, pos, edgelist=edges, width=3,alpha=0.5, edge_color="b", style="dashed")
        edge_labels = nx.get_edge_attributes(self.G, "weight")
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels)
        # Очистим выход
        self.output_frame.delete('1.0', END)

        # Отобразим матрицу смежности в выходном поле
        self.output_frame.insert(END, 'Матрица смежности:\n')
        self.output_frame.insert(END, np.array2string(self.nparr))

        # Вставим нарисованную фигуру в рамку вывода
        self.canvas = FigureCanvasTkAgg(fig, master=self.output_frame)
        
        self.canvas.draw()
        self.canvas.get_tk_widget().pack()

    def restart_program(self):
        python = sys.executable
        os.execl(python, python, *sys.argv)

    def dfs(self):
        n = self.nparr.shape[0]  # Количество вершин в графе
        if self.vert not in range(n):
            self.vert = 0
        visited = np.zeros(n, dtype=bool)  # Инициализируем массив для отслеживания посещенных вершин
        traversal = []  # Инициализируем список для сохранения порядка обхода DFS
        a = list(range(n))
        a.remove(self.vert)
        a[0] = self.vert

        # Выполним DFS, начиная с заданной вершины
        def dfs_recursive(vertex):
            visited[vertex] = True
            traversal.append(vertex)

            # Найдем индексы соседних вершин, используя матрицу смежности
            neighbors = np.where(self.nparr[vertex] == 1)[0]
            neighbors = neighbors[::-1]
            # Рекурсивно посетим все соседние
            for neighbor in neighbors:
                if not visited[neighbor]:
                    dfs_recursive(neighbor)

        # Выполните DFS для каждой непосещенной вершины
        for vertex in a:
            if not visited[vertex]:
                dfs_recursive(vertex)
        self.text_2.delete('1.0', END)
        self.text_2.insert('insert', f'Длина маршрута в глубину-{len(traversal)} :' + str(traversal) + '\n')

    def bfs(self):
        n = self.nparr.shape[0]  # Количество вершин в графе
        if self.vert not in range(n):
            self.vert = 0
        visited = np.zeros(n, dtype=bool)  # Инициализируем массив для отслеживания посещенных вершин
        traversal = []  # Инициализируем список для сохранения порядка обхода BFS
        a = list(range(n))
        a.remove(self.vert)
        a[0] = self.vert

# Выполним BFS, начиная с заданной вершины
        def bfs_start(vertex):
            queue = deque()
            queue.append(vertex)
            visited[vertex] = True

            while queue:
                current_vertex = queue.popleft()
                traversal.append(current_vertex)

                # Найдем индексы соседних вершин, используя матрицу смежности
                neighbors = np.where(self.nparr[current_vertex] == 1)[0]

                # Поставим в очередь непрошеных соседей
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        queue.append(neighbor)
                        visited[neighbor] = True

        # Выполним BFS для каждой непосещенной вершины
        for vertex in a:
            if not visited[vertex]:
                bfs_start(vertex)
        self.text_2.delete('1.0', END)
        self.text_2.insert('insert', f'Длина маршрута в ширину-{len(traversal)} :' + str(traversal) + '\n')

    def get_link_v(self, v):
        a=[]
        for i, weight in enumerate(self.adjacency_matrix[v]):
            if weight>0:
                a.append(i)
        return a
    
    def arg_min(self, T, S):
        amin=-1
        m=max(T)
        for i,t in enumerate(T):
            if t<m and i not in S:
                m=t
                amin=i
        return amin
    
    def dijxtra(self, start, end):
        N = len(self.adjacency_matrix)
        T = [math.inf]*N
        visited = {start}
        T[start]=0
        while start!=-1:
            for j in self.get_link_v(start):
                if j not in visited:
                    weight=T[j]+self.adjacency_matrix[start][j]
                    #print(weight)
                    if weight<T[j]:
                        T[j]=weight
            start=self.arg_min(T,visited)
            print(start)
            if start>0:
                visited.add(start)
        print(visited)
        self.text_2.delete('1.0', END)
        self.text_2.insert('insert', np.array2string(np.array(T)))

    def floyd(self):
        def get_path(P, u, v):
            path = [u]
            while u != v:
                u = P[u][v]
                path.append(u)

            return path
        N = len(self.adjacency_matrix)
        matr=self.adjacency_matrix
        for i in range(N):
            for j in range(N):
                if matr[i][j]==0:
                    matr[i][j]=math.inf
        P = [[v for v in range(N)] for u in range(N)]       # начальный список предыдущих вершин для поиска кратчайших маршрутов
        for k in range(N):
            for i in range(N):
                for j in range(N):
                    d = matr[i][k] + matr[k][j]
                    if matr[i][j] > d:
                        matr[i][j] = d
                        P[i][j] = k    
            print(P)
        
        start = int(self.e1.get().replace(' ', '').split(',')[0])
        end = int(self.e1.get().replace(' ', '').split(',')[1])
        path = get_path(P, end, start)
        path.reverse()
        print(path)
        self.text_2.delete('1.0', END)
        self.text_2.insert('insert', str('->'.join(str(x) for x in path)))

if __name__ == "__main__":
    app = GraphApp()
    app.run()