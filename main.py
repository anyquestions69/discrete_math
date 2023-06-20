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
        self.window.geometry('800x720+420+110')
        self.window.title('Графы')
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
        self.end = 3
        self.adjacency_matrix=[[0,2,3,0,20], 
                               [2,0,5,3,0], 
                               [3,5,0,1,7], 
                               [0,3,1,0,7],
                               [20,0,7,7,0]] 

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
        text = self.text_1.get('1.0', 'end-1c').replace(' ', '')
        self.adjacency_matrix=[]
        for i in text.split('\n'):
            row=[]
            for j in i.split(','):
                row.append(int(j))
            self.adjacency_matrix.append(row)
        self.adjacency_matrix = np.matrix([[int(j) for j in i] for i in text.split('\n')])
        self.nparr= np.array(self.adjacency_matrix)
        # Создим сетевой график
        self.G = nx.from_numpy_array(self.nparr, create_using=nx.DiGraph())
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
            neighbors = np.where(self.nparr[vertex] != 0)[0]
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
        self.text_2.insert('insert', f'Длина маршрута в глубину - {len(traversal)} :' + str(traversal) + '\n')

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
                neighbors = np.where(self.nparr[current_vertex] != 0)[0]

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
        self.text_2.insert('insert', f'Длина маршрута в ширину - {len(traversal)} :' + str(traversal) + '\n')

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
        v = 0       # стартовая вершина (нумерация с нуля)
        s = {v}     # просмотренные вершины
        t[v] = 0    # нулевой вес для стартовой вершины
        m = [0]*n   # оптимальные связи между вершинами
        matr=self.adjacency_matrix
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
                        

            v = self.arg_min(t, s) 
            
            if v >= 0:                    # выбрана очередная вершина
                s.add(v)        
        start = int(self.e1.get().replace(' ', '').split(',')[0])
        end = int(self.e1.get().replace(' ', '').split(',')[1])
        longiness = t[end]
        p = [end]
        while end != start:
            end = m[p[-1]]
            p.append(end)

        p.reverse()
        self.text_2.delete('1.0', END)
        self.text_2.insert('insert', np.array2string(np.array(p)))
        self.text_2.insert('insert', '\nДлина пути - '+str(longiness))

   
        
    def floyd(self):
        n = len(self.adjacency_matrix)
        start = int(self.e1.get().replace(' ', '').split(',')[0])
        end = int(self.e1.get().replace(' ', '').split(',')[1])
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
       

    def ford(self):
        def get_max_vertex(k, V, S):
            m = 0   # наименьшее допустимое значение
            v = -1
            for i, w in enumerate(V[k]):
                if i in S:
                    continue

            if w[2] == 1:   # движение по стрелке
                if m < w[0]:
                    m = w[0]
                    v = i
            else:           # движение против стрелки
                if m < w[1]:
                    m = w[1]
                    v = i

            return v


        def get_max_flow(T):
            w = [x[0] for x in T]
            return min(*w)


        def updateV(V, T, f):
            for t in T:
                if t[1] == -1:  # это исток
                    continue

                sgn = V[t[2]][t[1]][2]  # направление движения

                # меняем веса в таблице для (i,j) и (j,i)
                V[t[1]][t[2]][0] -= f * sgn
                V[t[1]][t[2]][1] += f * sgn

                V[t[2]][t[1]][0] -= f * sgn
                V[t[2]][t[1]][1] += f * sgn


        V = [[[0,0,1], [20,0,1], [30,0,1], [10,0,1], [0,0,1]],
            [[20,0,-1], [0,0,1], [40,0,1], [0,0,1], [30,0,1]],
            [[30,0,-1], [40,0,-1], [0,0,1], [10,0,1], [20,0,1]],
            [[10,0,-1], [0,0,1], [10,0,-1], [0,0,1], [20,0,1]],
            [[0,0,1], [30,0,-1], [20,0,-1], [20,0,-1], [0,0,1]],
        ]

        N = len(V)  # число вершин в графе
        init = 0    # вершина истока (нумерация с нуля)
        end = 4     # вершина стока
        Tinit = (math.inf, -1, init)      # первая метка маршруто (a, from, vertex)
        f = []      # максимальные потоки найденных маршрутов

        j = init
        while j != -1:
            k = init  # стартовая вершина (нумерация с нуля)
            T = [Tinit]  # метки маршрута
            S = {init}  # множество просмотренных вершин

            while k != end:     # пока не дошли до стока
                j = get_max_vertex(k, V, S)  # выбираем вершину с наибольшей пропускной способностью
                if j == -1:         # если следующих вершин нет
                    if k == init:      # и мы на истоке, то
                        break          # завершаем поиск маршрутов
                    else:           # иначе, переходим к предыдущей вершине
                        k = T.pop()[2]
                        continue

                c = V[k][j][0] if V[k][j][2] == 1 else V[k][j][1]   # определяем текущий поток
                T.append((c, j, k))    # добавляем метку маршрута
                S.add(j)            # запоминаем вершину как просмотренную

                if j == end:    # если дошди до стока
                    f.append(get_max_flow(T))     # находим максимальную пропускную способность маршрута
                    updateV(V, T, f[-1])        # обновляем веса дуг
                    break

                k = j

        F = sum(f)
        print(f"Максимальный поток равен: {F}")

if __name__ == "__main__":
    app = GraphApp()
    app.run()