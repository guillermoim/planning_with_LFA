

(define (problem BW-rand-5)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5)
(:init
(arm-empty)
(on-table b4)
(on b1 b4)
(on b5 b2)
(on b3 b5)
(on-table b2)
(clear b3)
(clear b1)
)
(:goal
(and
(arm-empty)
(on b4 b3))
)
)


