from tkinter import *
#import networkx as nx
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
        self.window.geometry('600x520+320+90')
        self.window.title('Графы')
        self.window.config(bg='#1a1a1a')
        self.label1= Label(self.window, text='Введите матрицу, разделяя элементы запятой, а строки новой строкой')
        self.label1.place(x=10, y=10)
        self.text_1 = Text(self.window, font=('Arial', 15, 'bold'))
        self.text_1.tag_configure("center", justify='center')
        self.text_1.place(x=10, y=40, width=290, height=100)
        self.label2= Label(self.window, text='Вывод')
        self.label2.place(x=10, y=160)
        self.label3= Label(self.window, text='Выбор алгоритма')
        self.label3.place(x=320, y=160)
        self.output_frame = Text(self.window, bg='white', foreground='black')
        self.output_frame.place(x=10, y=180, width=290, height=200)

        self.text_2 = Text(self.window, font=('Arial', 13, 'bold'),foreground='black',bg='white')
        self.text_2.tag_configure("center", justify='center')
        self.text_2.place(x=10, y=390, width=290, height=80)

        self.btn_1 = Button(self.window, text='Построить граф', command=self.display_graph)
        self.btn_1.place(x=320, y=40, width=110, height=50)

        self.btn_2 = Button(self.window, text='Очистить', command=self.restart_program)
        self.btn_2.place(x=430, y=40, width=110, height=50)

        self.btn_3 = Button(self.window, text='Обход в глубину', command=self.dfs)
        self.btn_3.place(x=320, y=230, width=110, height=50)

        self.btn_4 = Button(self.window, text='Обход в ширину', command=self.bfs)
        self.btn_4.place(x=430, y=230, width=110, height=50)

        self.text_3 = Text(self.window, font=('Arial', 13, 'bold'))
        self.text_3.tag_configure("center", justify='center')
        self.text_3.place(x=320, y=180, width=110, height=50)

        self.btn_5 = Button(self.window, text='Ввод вершины', command=self.vertex)
        self.btn_5.place(x=430, y=180, width=110, height=50)

        self.btn_6 = Button(self.window, text='Дейкстры', command=self.dijxtra)
        self.btn_6.place(x=320, y=370, width=110, height=50)

        self.btn_7 = Button(self.window, text='Флойда', command=self.floyd)
        self.btn_7.place(x=430, y=370, width=110, height=50)

        self.btn_8 = Button(self.window, text='Форда-Фалкерсона', command=self.ford)
        self.btn_8.place(x=485, y=420, width=110, height=50)

        """  self.text_4 = Text(self.window, font=('Arial', 13, 'bold'))
        self.text_4.tag_configure("center", justify='center')
        self.text_4.place(x=10, y=620, width=400, height=80) """

        self.e1 = Entry(self.window)#Text(self.window, font=('Arial', 15, 'bold'))
        self.e1.insert(0, '1,4')
        self.e1.place(x=320, y=320, width=220, height=50)
        self.vert = 0
        self.start=0
        self.end = 3
        self.adjacency_matrix=[[0,7,9,0,0,0], 
                               [0,0,10,0,0,0], 
                               [0,0,0,11,0,2], 
                               [0,15,0,0,0,0],
                               [0,0,0,6,0,0],
                               [14,0,0,0,9,0]] 

    def vertex(self):
        self.vert = int(self.text_3.get('1.0', 'end-1c').replace(' ', ''))-1

    def run(self):
        self.window.mainloop()

    def display_graph(self):
        try: 
            self.canvas.get_tk_widget().pack_forget()
        except AttributeError: 
            pass                
        self.output_frame.delete('1.0', END)
        text = self.text_1.get('1.0', 'end-1c')
        if text!='':
            self.adjacency_matrix=[]
            for i in text.split('\n'):
                row=[]
                for j in i.split(' '):
                    row.append(int(j))
                self.adjacency_matrix.append(row) 
        #self.adjacency_matrix = np.matrix([[int(j) for j in i] for i in text.split('\n')])
        self.nparr= np.array(self.adjacency_matrix)
        
        # Очистим выход
        self.output_frame.delete('1.0', END)

        # Отобразим матрицу смежности в выходном поле
        self.output_frame.insert(END, 'Матрица весов:\n')
        self.output_frame.insert(END, np.array2string(self.nparr))


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
        spanning_tree={}
        self.tree=[]
        for i in range(len(self.adjacency_matrix)):
            self.tree.append([0]*len(self.adjacency_matrix))
        
        # Выполним DFS, начиная с заданной вершины
        def dfs_recursive(vertex):
            visited[vertex] = True
            traversal.append(vertex)

            # Найдем индексы соседних вершин, используя матрицу смежности
            neighbors = np.where(self.nparr[vertex] != 0)[0]
            #neighbors = neighbors[::-1]
            # Рекурсивно посетим все соседние
            for neighbor in neighbors:
                if not visited[neighbor]:
                    dfs_recursive(neighbor)
                    if vertex not in spanning_tree.keys():
                        spanning_tree[vertex] = []
                    if neighbor not in spanning_tree.keys():
                        spanning_tree[neighbor] = []
                    spanning_tree[vertex].append(neighbor)
                    
                    
        # Выполните DFS для каждой непосещенной вершины
        for vertex in a:
            if not visited[vertex]:
                dfs_recursive(vertex)
        print(spanning_tree)
        for key,value in spanning_tree.items():
            for v in value:
                if key in range(len(self.tree)) and v  in range(len(self.tree)) :
                    self.tree[key][v]=1
        for i in range(len(traversal)):
            traversal[i]+=1
        self.text_2.delete('1.0', END)
        self.text_2.insert('insert', f'Длина маршрута в глубину - {len(traversal)} :' + str(traversal) + '\n')
        self.output_frame.insert('insert', f'\n\nМатрица смежности покрывающего дерева dfs:\n ' + np.array2string(np.array(self.tree)) + '\n')

   
    def bfs(self):
        n = self.nparr.shape[0]  # Количество вершин в графе
        if self.vert not in range(n+1):
            self.vert = 0
        visited = np.zeros(n, dtype=bool)  # Инициализируем массив для отслеживания посещенных вершин
        traversal = []  # Инициализируем список для сохранения порядка обхода BFS
        a = list(range(n))
        a.remove(self.vert)
        a[0] = self.vert
        self.tree=[]
        for i in range(len(self.adjacency_matrix)):
            self.tree.append([0]*len(self.adjacency_matrix))
        spanning_tree={}
# Выполним BFS, начиная с заданной вершины
        def bfs_start(vertex):
            queue = deque()
            queue.append(vertex)
            visited[vertex] = True

            while queue:
                current_vertex = queue.popleft()
                traversal.append(current_vertex)

                # Найдем индексы соседних вершин, используя матрицу смежности
                neighbors = np.where(self.nparr[current_vertex] != 0)[0]

                # Поставим в очередь непрошеных соседей
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        queue.append(neighbor)
                        if current_vertex not in spanning_tree.keys():
                            spanning_tree[current_vertex] = []
                        if neighbor not in spanning_tree.keys():
                            spanning_tree[neighbor] = []
                        spanning_tree[current_vertex].append(neighbor)
                        #spanning_tree[neighbor].append(current_vertex)
                        visited[neighbor] = True
            print(spanning_tree)
            for key,value in spanning_tree.items():
                for v in value:
                    if key in range(len(self.tree)) and v  in range(len(self.tree)) :
                        self.tree[key][v]=1

        # Выполним BFS для каждой непосещенной вершины
        for vertex in a:
            if not visited[vertex]:
                bfs_start(vertex)

        for i in range(len(traversal)):
            traversal[i]+=1
        self.text_2.delete('1.0', END)
        self.text_2.insert('insert', f'Длина маршрута в ширину - {len(traversal)} :' + str(traversal) + '\n')
        self.output_frame.insert('insert', f'\n\nМатрица смежности покрывающего дерева bfs:\n ' + np.array2string(np.array(self.tree)) + '\n')
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
    
    def dijxtra(self):
        n = len(self.adjacency_matrix)
        t = [math.inf]*n   # последняя строка таблицы
        matr=[]
        v = int(self.e1.get().replace(' ', '').split(',')[0])-1       # стартовая вершина (нумерация с нуля)
        s = {v}     # просмотренные вершины
        t[v] = 0    # нулевой вес для стартовой вершины
        m = [0]*n   # оптимальные связи между вершинами
        matr=self.adjacency_matrix
        spanning_tree={}
        tree=[]
        for i in range(len(self.adjacency_matrix)):
            tree.append([0]*len(self.adjacency_matrix))
        longiness=0
        for i in range(n):
            for j in range(n):
                if matr[i][j]==0:
                    matr[i][j]=math.inf
        while v != -1:         
            for j, dw in enumerate(matr[v]): 
                  
                if j not in s:  
                    w = t[v] + dw
                    if w < t[j]:
                        t[j] = w
                        m[j] = v
                        
            #spanning_tree[v]= self.arg_min(t, s)
                        
            v = self.arg_min(t, s) 
            
            if v >= 0:                    # выбрана очередная вершина
                s.add(v)  
                
         
        start = int(self.e1.get().replace(' ', '').split(',')[0])-1
        end = int(self.e1.get().replace(' ', '').split(',')[1])-1
        longiness = t[end]
        p = [end]
        while end != start:
            end = m[p[-1]]
            p.append(end)

        p.reverse()
        for i in range(len(p)-1):
            spanning_tree[p[i]]=p[i+1]
        print(spanning_tree)   
        for key,value in spanning_tree.items():
                if key in range(len(tree)) and value  in range(len(tree)) :
                    tree[key][value]=1

        print(tree)  
        for i in range(len(p)):
            p[i]+=1
        self.text_2.delete('1.0', END)
        self.text_2.insert('insert', np.array2string(np.array(p)))
        self.text_2.insert('insert', '\nДлина пути - '+str(longiness))
        self.output_frame.insert('insert', f'\n\nМатрица смежности покрывающего дерева :\n ' + np.array2string(np.array(tree)) + '\n')


   
        
    def floyd(self):
        n = len(self.adjacency_matrix)
        start = int(self.e1.get().replace(' ', '').split(',')[0])-1
        end = int(self.e1.get().replace(' ', '').split(',')[1])-1
        s={start}
        matrix=self.adjacency_matrix
        for i in range(n):
            for j in range(n):
                if matrix[i][j]==0:
                    matrix[i][j]=math.inf
                else:
                    matrix[i][j]=self.adjacency_matrix[i][j]
        for k in range(len(matrix)): 
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    if i==start and j==end:
                        if matrix[i][j]< matrix[i][k] + matrix[k][j]:
                            s.add(j)
                        else:
                            s.add(k)
                    matrix[i][j] = min(matrix[i][j], matrix[i][k] + matrix[k][j])
        self.text_2.delete('1.0', END)
        print(s)
        longiness =matrix[start][end]
        self.text_2.insert('insert', 'Длина пути - '+str(longiness)+'\n')
        #self.text_2.insert('insert', 'Пути - '+str(s)+'\n')
        self.text_2.insert('insert', str('\n'.join(str(x) for x in matrix)))
        return matrix
       

    def ford(self, source=0, sink=None):
        if sink is None:
            sink = self.nparr.shape[0] - 1
        n = len(self.adjacency_matrix)
        residual = [[self.adjacency_matrix[i][j] for j in range(n)] for i in range(n)]
        flow = [[0 for _ in range(n)] for _ in range(n)]
        parents = [-1] * n  # Список для отслеживания родителей вершин в найденных путях

        def bfs():
            queue = [source]
            visited = [False] * n
            visited[source] = True

            while queue:
                u = queue.pop(0)
                for v in range(n):
                    if not visited[v] and residual[u][v] > 0:
                        queue.append(v)
                        visited[v] = True
                        parents[v] = u
                        if v == sink:
                            return True  # Путь до стока найден
            return False

        max_flow = 0

        while bfs():
            path_flow = float('inf')
            v = sink
            while v != source:
                u = parents[v]
                path_flow = min(path_flow, residual[u][v])
                v = u

            v = sink
            while v != source:
                u = parents[v]
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                flow[u][v] += path_flow
                v = u

            max_flow += path_flow

        # text = '['
        # for row in flow:
        #     text += str(row) +',\n'
        # text = text[:-2] + ']\n'
        self.text_2.delete('1.0', END)
        self.text_2.insert('insert',f'\n\n[Максимальный путь: {str(max_flow)}]')


if __name__ == "__main__":
    app = GraphApp()
    app.run()