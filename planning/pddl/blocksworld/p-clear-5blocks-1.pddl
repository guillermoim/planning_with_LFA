

(define (problem BW-rand-5)
(:domain blocksworld)
(:objects b1 b2 b3 b4 b5 )
(:init
(arm-empty)
(on-table b4)
(on b1 b4)
(on b3 b1)
(on b5 b2)
(on-table b2)
(clear b3)
(clear b5)
)
(:goal
(and
(arm-empty)
(clear b2))
)
)