(define (problem BW-rand-3)
(:domain blocksworld)
(:objects b1 b2 b3)
(:init
(arm-empty)
(on-table b2)
(on b1 b2)
(on b3 b1)
(clear b3)
)
(:goal
(and
(arm-empty)
(clear b2))
)
)