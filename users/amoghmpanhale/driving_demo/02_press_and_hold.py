import pygame

pygame.init()

screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        print("W key is being held down")
    if keys[pygame.K_s]:
        print("S key is being held down")
    if keys[pygame.K_a]:
        print("A key is being held down")
    if keys[pygame.K_d]:
        print("D key is being held down")

    screen.fill((0, 0, 0))
    pygame.display.flip()
    clock.tick(60)