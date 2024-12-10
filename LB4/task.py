import cv2
import numpy as np


def GAUSSBlur(filepath: str, blur_kernel_size: int):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    return cv2.GaussianBlur(small, (blur_kernel_size, blur_kernel_size), cv2.BORDER_DEFAULT)

def filter(img, ker):
    #B = img.copy()
    B=np.zeros_like(img, np.int32)
    x0 = ker.shape[0] // 2
    y0 = ker.shape[1] // 2
    size_ker=int(ker.shape[0]//2)
    for i in range(x0, B.shape[0] - x0):
        for j in range(y0, B.shape[1] - y0):
           val = 0

           for k in range(-size_ker, size_ker + 1):
                for l in range(-size_ker, size_ker+ 1):
                    val += img[i + k, j + l] * ker[k + size_ker, l + size_ker]
                B[i, j] = val

    return B



def angle_value(x, y, tang):
    if (x >= 0 and y <= 0 and tang < -2.414) or (x <= 0 and y <= 0 and tang > 2.414):
        return 0  # сер
    elif x >= 0 and y <= 0 and tang < -0.414:
        return 1  # корич
    elif (x >= 0 and y <= 0 and tang > -0.414) or (x >= 0 and y >= 0 and tang < 0.414):
        return 2  # крас
    elif x >= 0 and y >= 0 and tang < 2.414:
        return 3  # оранж
    elif (x >= 0 and y >= 0 and tang > 2.414) or (x <= 0 and y >= 0 and tang < -2.414):
        return 4  # желт
    elif x <= 0 and y >= 0 and tang < -0.414:
        return 5  # зел
    elif (x <= 0 and y >= 0 and tang > -0.414) or (x <= 0 and y <= 0 and tang < 0.414):
        return 6  # голуб
    elif x <= 0 and y <= 0 and tang < 2.414:
        return 7  # син


def edge_detection(grayscale_image, lower_bar, high_bar):
    # Определение матриц Собеля для вычисления градиентов по горизонтали и вертикали
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # применяем свертку для получения градиентов
    Gx = filter(grayscale_image, sobel_x)
    Gy = filter(grayscale_image, sobel_y)
    Gx_safe = np.where(Gx == 0, 1e-10, Gx)
    # длины и углы градиентов
    grad_len = np.sqrt(np.add(np.square(Gx), np.square(Gy)))  # длина вектора град = (gx^2+gy^2)^0.5
    G = np.divide(Gy, Gx_safe)
    G = np.nan_to_num(G)  # Замена NaN на 0

    angle_print = np.arctan(G) * (180 / np.pi)

    print("Lenght gradient")
    print(grad_len)  # Вывод длины градиента
    print("Angle")
    print(angle_print)  # Вывод углов градиента

    # Подавление не максимумов (№3)
    edges = np.zeros_like(grayscale_image)  # создание пустого изображения для границ

    for y in range(1, edges.shape[0] - 1):  # проход по всем пикселям, не включая границы
        for x in range(1, edges.shape[1] - 1):
            line = angle_value(Gx[y, x], Gy[y, x], G[y, x])  # получение значения угла
            # Определение соседних пикселей в зависимости от угла
            if line == 0 or line == 4:
                neighbor1 = [y - 1, x]
                neighbor2 = [y + 1, x]
            elif line == 1 or line == 5:
                neighbor1 = [y - 1, x + 1]
                neighbor2 = [y + 1, x - 1]
            elif line == 2 or line == 6:
                neighbor1 = [y, x + 1]
                neighbor2 = [y, x - 1]
            elif line == 3 or line == 7:
                neighbor1 = [y + 1, x + 1]
                neighbor2 = [y - 1, x - 1]
            else:
                raise Exception('Angle not defined')
            if grad_len[y, x] >= grad_len[neighbor1[0], neighbor1[1]] and grad_len[y, x] > grad_len[
                neighbor2[0], neighbor2[1]]:  # Сравнение текущего градиента с соседними
                edges[y, x] = 255  # пометка пикселя как границы

    cv2.imshow('NON MAX task3', edges)

    # двойная пороговая фильтрация (№4)
    max_grad_len = grad_len.max()  # максимальная длина градиента
    low_level = int(max_grad_len * lower_bar)  # низкий порог
    high_level = int(max_grad_len * high_bar)  # высокий порог
    for y in range(edges.shape[0]):  # проход по всем пикселям
        for x in range(edges.shape[1]):
            if edges[y, x] > 0:  # если пиксель часть границы
                if grad_len[y, x] < low_level:
                    edges[y, x] = 0  # убираем, если градиент ниже низкого порога
                elif grad_len[y, x] < high_level:
                    keep = False  # флаг для хранения пикселя
                    # проверка соседей на высокий градиент
                    for neighbor_y in (y - 1, y, y + 1):
                        for neighbor_x in (x - 1, x, x + 1):
                            if neighbor_y != y or neighbor_x != x:
                                if edges[neighbor_y, neighbor_x] > 0 and grad_len[neighbor_y, neighbor_x] >= high_level:
                                    keep = True  # сохраняем, если соседний пиксель границ
                    if not keep:
                        edges[y, x] = 0  # убираем, если нет соседей с высоким градиентом

    return edges  # возвращаем изображение с границами


if __name__ == '__main__':

    image = 'img.jpg'
    img = cv2.imread(r'img.jpg', cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)


    blur = 5
    lower_bar = 0.1
    high_bar = 0.22

    image = GAUSSBlur(image, blur)

    edges = edge_detection(image, lower_bar, high_bar)

    cv2.imshow('GAUSS task1', image)
    cv2.imshow('DOUBLE FILTER task4', edges)

    cv2.waitKey(0)
    cv2.destroyAllWindows()