

(define (problem BW-rand-5)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 b6 b7)
(:init
(arm-empty)
(on-table b7)
(on b1 b4)
(on b6 b1)
(on b3 b6)
(clear b3)
(on b4 b7)
(on-table b2)
(on b5 b2)
(clear b5)
)
(:goal
(and
(arm-empty)
(clear b7)
)
)
)

