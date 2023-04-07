from PIL import Image
import numpy as np
import glob
import random
import tabulate
import time
import os
import colorthief
import pickle


def generating(n_objects_for_im, n_generations, n_objects_survive, n_objects_children, ob_min_size_per, out_of_frame,
               printing_ob_list, saving, reduce_pic_size):

    # Wypisanie parametrów
    def print_parameters(objects_set_number, n_objects_variations, n_objects_for_im, n_generations, n_objects_survive,
                         n_objects_children, ob_min_size_per, out_of_frame, reduce_pic_size, printing_ob_list, saving):
        print(f"Objects set's number: {objects_set_number}")
        print("-")
        print(f"Objects' variations: {n_objects_variations}")
        print(f"Objects for picture: {n_objects_for_im}")
        print(f"Generations: {n_generations}")
        print(f"Sum of objects of population in one generation: {n_objects_survive * (n_objects_children + 1)}")
        print(f"Percentage of objects that survive generation: {(n_objects_survive / (n_objects_survive * (n_objects_children + 1))) * 100}%")
        print("-")
        print(f"Minimum object size: {ob_min_size_per * 100}%")
        print(f"Out of frame: {out_of_frame * 100}%")
        print(f"Picture size division: {reduce_pic_size}")
        print(f"Showing objects parameters: {printing_ob_list}")
        print(f"Saving objects: {saving}")
        print("")

    # Generowanie jednego obiekty na podstawie jego parametrów
    def generate_one_object(ob_type, ob_size, ob_rotation, ob_position):
        ob = Image.open(f"objects_sets\\{objects_set_number}\\{ob_type}.png")
        ob_im = gen_im.copy()
        ob = ob.resize((ob_size, ob_size))

        ob = ob.rotate(ob_rotation, expand=True, fillcolor=(0, 0, 0, 0))
        ob_size_after = ob.size

        ob_pos = (round((pic_size[0] - (ob_size - 2 * round(out_of_frame * ob_size))) * ob_position[0] -
                 round(out_of_frame * ob_size)),
                 round((pic_size[1] - (ob_size - 2 * round(out_of_frame * ob_size))) * ob_position[1] -
                 round(out_of_frame * ob_size)))
        ob_pos_cor = (ob_pos[0] - (abs(ob_size - ob_size_after[0]) // 2),
                      ob_pos[1] - (abs(ob_size - ob_size_after[1]) // 2))

        # Automatyczny kolor (średnia)
        just_letter_im = Image.new(mode="RGBA", size=pic_size, color=(0, 0, 0, 0))
        ob = ob.convert("RGBA")
        just_letter_im.paste(ob, ob_pos_cor, ob)
        just_letter_im = Image.composite(pic, just_letter_im, just_letter_im)
        data_c = np.array(just_letter_im)
        _, _, _, alpha = data_c.T
        areas = (alpha == 255)
        mean_colour = np.round(np.mean(data_c[..., :-1][areas.T], axis=0))
        data_c = np.array(ob)
        red, green, blue, alpha = data_c.T
        areas = (red == 0) & (blue == 0) & (green == 0) & (alpha != 0)
        data_c[..., :-1][areas.T] = mean_colour
        ob = Image.fromarray(data_c)

        ob_im.paste(ob, ob_pos_cor, ob)

        return ob_im

    objects_set_number = 1  # Numer zestawu obiektów

    # Wgranie obrazu z foldera
    FPATH = ""
    try:
        FPATH = glob.glob("picture/*.jpg")[0]
    except:
        try:
            FPATH = glob.glob("picture/*.jpeg")[0]
        except:
            try:
                FPATH = glob.glob("picture/*.png")[0]
            except:
                print("Image file error.")
                quit()
    pic = Image.open(FPATH)
    pic = pic.resize((pic.size[0]//reduce_pic_size, pic.size[1]//reduce_pic_size))
    pic.show()
    pic.convert("RGBA")
    pic_size = pic.size
    pix_pic = pic.load()

    # Ustalenie przedziałów rozmiaru obiektu
    ob_min_size = int(max(pic_size[0], pic_size[1]) * ob_min_size_per) + 1
    ob_max_size = int(min(pic_size[0], pic_size[1]))

    # Usuwanie zapisanych obiektów oraz wyników
    for file in glob.glob("population/*.png"):
        os.remove(file)
    for file in glob.glob("generated_im/*.png"):
        os.remove(file)

    # Liczba dostępnych obiektów
    objects_files = glob.glob(f"objects_sets/{objects_set_number}/*.png")
    n_objects_variations = len(objects_files)

    # Wypisanie parametrów
    print_parameters(objects_set_number, n_objects_variations, n_objects_for_im, n_generations, n_objects_survive,
                     n_objects_children, ob_min_size_per, out_of_frame, reduce_pic_size, printing_ob_list, saving)

    # Stworzenie obrazu (tła), na który będą wklejane obiekty
    gen_im = Image.new(mode="RGBA", size=pic_size, color=colorthief.ColorThief(FPATH).get_color(quality=1))

    # Lista z finalnie dodanymi obiektami
    final_objects_list = np.empty((n_objects_for_im, 6), dtype=object)
    headers = ["n", "object", "size", "rotation", "position", "dif"]

    # Zapisanie parametrów danego uruchomienia
    run_params = {"objects_set_number": objects_set_number,
                  "n_objects_for_im": n_objects_for_im,
                  "n_generations": n_generations,
                  "n_objects_survive": n_objects_survive,
                  "n_objects_children": n_objects_children,
                  "ob_min_size_per": ob_min_size_per,
                  "out_of_frame": out_of_frame,
                  "reduce_pic_size": reduce_pic_size,
                  "pic_size": pic_size,
                  }

    with open("generated_im/run_params", "wb") as f:
        pickle.dump(run_params, f)

    for h in range(n_objects_for_im):

        # Lista z populacją obiektów
        population = np.empty((n_objects_survive * (n_objects_children + 1), 6), dtype=object)

        dif_array_gen = int(np.sum(np.abs(np.array(gen_im)[:, :, :3] - np.array(pic)[:, :, :3])))

        for generation in range(n_generations):

            # Ustalanie parametrów obiektów (który, rozmiar, rotacja, położenie)

            start_t = time.time()
            for i in range(n_objects_survive):
                for j in range(n_objects_children + 1):

                    # Generowanie populacji początkowej
                    if generation == 0:
                        i_ob = i * (n_objects_children + 1) + j

                        # Wczytanie i ustalenie obiektu
                        population[i_ob, 0] = i_ob + 1
                        object_file = random.choice(objects_files)
                        population[i_ob, 1] = str(object_file).replace(f"objects_sets/{objects_set_number}\\", "").replace(".png", "")

                        # Ustalenie rozmiaru
                        population[i_ob, 2] = random.randrange(ob_min_size, ob_max_size, 1)

                        # Ustalenie rotacji
                        population[i_ob, 3] = random.randrange(-360, 360, 1)

                        # Ustalenie położenia
                        population[i_ob, 4] = (round(random.random(), 3),
                                               round(random.random(), 3))

                    # Generowanie kolejnych populacji
                    else:
                        i_ob = i * n_objects_children + n_objects_survive + j
                        if j == n_objects_children:
                            break

                        # Wczytanie i ustalenie obiektu
                        population[i_ob, 1] = population[i, 1]

                        # Ustalenie zmiany rozmiaru
                        population[i_ob, 2] = population[i, 2] + round(np.random.normal(0, 0.1) * (ob_max_size - ob_min_size))
                        if population[i_ob, 2] < ob_min_size:
                            population[i_ob, 2] = ob_min_size
                        if population[i_ob, 2] > ob_max_size:
                            population[i_ob, 2] = ob_max_size

                        # Ustalenie zmiany rotacji
                        population[i_ob, 3] = population[i, 3] + round(np.random.normal(0, 0.1) * 360)
                        if population[i_ob, 3] > 360:
                            population[i_ob, 3] -= 360
                        if population[i_ob, 3] < 0:
                            population[i_ob, 3] += 360

                        # Ustalenie zmiany położenia
                        population[i_ob, 4] = [population[i, 4][0] + round(np.random.normal(0, 0.1) * 1, 3),
                                               population[i, 4][1] + round(np.random.normal(0, 0.1) * 1, 3)]
                        if population[i_ob, 4][0] < 0:
                            population[i_ob, 4][0] = 0
                        if population[i_ob, 4][0] > 1:
                            population[i_ob, 4][0] = 1
                        if population[i_ob, 4][1] < 0:
                            population[i_ob, 4][1] = 0
                        if population[i_ob, 4][1] > 1:
                            population[i_ob, 4][1] = 1

            print(f"Generating {generation + 1}. population for {h + 1}. object done in {round(time.time() - start_t, 2)} s.")

            # Mierzenie różnicy oryginału i z wklejonym obiektem

            start_t = time.time()
            for i in range(n_objects_survive * (n_objects_children + 1)):

                # Wygenerowanie obiektu na podstawie ustalonych parametrów
                ob_im = generate_one_object(population[i, 1], population[i, 2], population[i, 3], population[i, 4])

                # Zapis do folderu
                if saving:
                    sav_ob_im = ob_im.copy()
                    sav_ob_im = sav_ob_im.resize((pic_size[0] // 3, pic_size[1] // 3))
                    sav_ob_im.save(f"population/{population[i, 0]}.png", quality=50, optimize=True)

                # Zmierzenie różnicy obecnie wygenerowanego obrazu (+ wklejony obiekt) z podanym obrazem
                dif_array_ob = int(np.sum(np.abs(np.array(ob_im)[:, :, :3] - np.array(pic)[:, :, :3])))
                population[i, 5] = (dif_array_gen - dif_array_ob) * (-1)

            # Posortowanie obiektów według różnicy
            population = population[population[:, 5].argsort()]
            if printing_ob_list:
                print("\n", tabulate.tabulate(population, headers=headers), "\n")

            print(f"Calculating difference done in {round(time.time() - start_t, 2)} s.")
            print(f"Sum of difference at population: {round(sum(population[:, 5]), 3)}")
            print("")

            # Pozostawienie najbardziej dostosowanych obiektów
            population[n_objects_survive:, 1:] = None

        print(f"BEST {h + 1}. OBJECT: {population[0, 0]}")
        print("")

        # Zapisywanie parametrów wybranego najlepszego obiektu
        final_objects_list[h] = population[0].copy()
        with open("generated_im/objects_list", "wb") as f:
            pickle.dump(final_objects_list, f)

        # Wygenerowanie i wklejenie na generowany obrazek wybranego obiektu
        gen_im = generate_one_object(population[0, 1], population[0, 2], population[0, 3], population[0, 4])
        gen_im.save(f"generated_im/{h + 1}.png")


if __name__ == "__main__":

    # Ustalenie parametrów generowania

    n_objects_for_im = 1000  # Liczba obiektów finalnie tworzących wygenerowany obraz
    n_generations = 20  # Liczba pokoleń/iteracji algorytmu
    n_objects_survive = 200  # Liczba obiektów, która nie "wymiera" po iteracji algorytmu
    n_objects_children = 4  # Liczba potomstwa obiektu
    # Suma obiektów w "środowisku" w jednym pokoleniu to objects_survive * (objects_children + 1)
    ob_min_size_per = 0.006  # Parametr stanowiący dolny limit rozmiaru obiektów
    out_of_frame = 0.3  # Parametr wychodzenia obiektu poza obszar obrazu (Uwaga, obecnie za niski powoduje błąd)
    printing_ob_list = False  # Decyzja o wyświetleniu parametrów wygenerowanych obiektów
    saving = False  # Decyzja o zapisie wszystkich wygenerowanych wariacji obiektów do pliku
    reduce_pic_size = 3  # Dzielnik rozmiaru obrazu

    generating(n_objects_for_im, n_generations, n_objects_survive, n_objects_children, ob_min_size_per, out_of_frame,
               printing_ob_list, saving, reduce_pic_size)
