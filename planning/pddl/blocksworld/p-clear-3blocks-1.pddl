(define (problem BW-rand-3)
(:domain blocksworld)
(:objects b1 b2 b3)
(:init
(arm-empty)
(on-table b1)
(on b2 b1)
(on-table b3)
(clear b3)
(clear b2)
)
(:goal
(and
(arm-empty)
(clear b1))
)
)